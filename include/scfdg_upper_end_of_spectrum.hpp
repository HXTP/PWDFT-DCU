/*
   Copyright (c) 2017 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Amartya Banerjee

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
/// @file scfdg_upper_end_of_spectrum.hpp

#ifndef _SCFDG_UPPER_END_SPECTRUM_HPP_
#define _SCFDG_UPPER_END_SPECTRUM_HPP_

namespace dgdft{
  
// This should be called only by prcessors sharing the context of Hmat
double find_comp_subspace_UB_parallel(dgdft::scalapack::ScaLAPACKMatrix<Real>& Hmat)
{
    double b_up = 0.0;  
	double alpha, beta;
	double minus_alpha, minus_beta;

	char uplo = 'N';
	const double scalar_one = 1.0;
	const double scalar_minus_one = -1.0;
	const double scalar_zero = 0.0;
			
	const int ht = Hmat.Height();     
	const int context = Hmat.Context();
	const int scaBlockSize = Hmat.MB();

	// Set up v0
	dgdft::scalapack::Descriptor vec_v0_desc;
	vec_v0_desc.Init( ht, 1, 
			scaBlockSize, scaBlockSize, 
			0, 0, 
			context);   
	dgdft::scalapack::ScaLAPACKMatrix<Real>  vec_v0;
	vec_v0.SetDescriptor(vec_v0_desc);

	// Set up v
	dgdft::scalapack::Descriptor vec_v_desc;
	vec_v_desc.Init( ht, 1, 
			scaBlockSize, scaBlockSize, 
			0, 0, 
			context);   
	dgdft::scalapack::ScaLAPACKMatrix<Real>  vec_v;     
	vec_v.SetDescriptor(vec_v_desc);
	
	// Set up f
	dgdft::scalapack::Descriptor vec_f_desc;
	vec_f_desc.Init( ht, 1, 
			scaBlockSize, scaBlockSize, 
			0, 0, 
			context);   
	dgdft::scalapack::ScaLAPACKMatrix<Real>  vec_f;    
	vec_f.SetDescriptor(vec_f_desc);
	
	// Randomly initialize vector v
	double *data_ptr =  vec_v.Data();    
	for (int ii = 0; ii < vec_v0.LocalHeight(); ii ++)
	data_ptr[ii] = UniformRandom();
	
	// Normalize this vector
	double nrm;     
	dgdft::scalapack::SCALAPACK(pdnrm2)(&ht, &nrm, vec_v.Data(), &I_ONE, &I_ONE, vec_v.Desc().Values(), &I_ONE);

	double scalar_a = 1.0 / nrm;
	dgdft::scalapack::SCALAPACK(pdscal)(&ht , &scalar_a , vec_v.Data() , &I_ONE , &I_ONE , vec_v.Desc().Values(), &I_ONE);

	// Compute f = H * v : use -H here 
	dgdft::scalapack::SCALAPACK(pdgemv)(&uplo , &ht , &ht , &scalar_minus_one , Hmat.Data() , &I_ONE , &I_ONE , Hmat.Desc().Values() , 
					vec_v.Data() , &I_ONE , &I_ONE , vec_v.Desc().Values() , &I_ONE , 
					&scalar_zero , vec_f.Data(), &I_ONE , &I_ONE , vec_f.Desc().Values() , &I_ONE);

	// alpha = dot(f,v)
	dgdft::scalapack::SCALAPACK(pddot)(&ht , &alpha ,  vec_f.Data(), &I_ONE , &I_ONE , vec_f.Desc().Values() , &I_ONE , 
							vec_v.Data() ,  &I_ONE , &I_ONE , vec_v.Desc().Values() , &I_ONE );
	
	// f = f - alpha * v;
	minus_alpha = -alpha;
	dgdft::scalapack::SCALAPACK(pdaxpy)(&ht, &minus_alpha , vec_v.Data(), &I_ONE , &I_ONE , vec_v.Desc().Values() , &I_ONE , 
								vec_f.Data() , &I_ONE , &I_ONE , vec_f.Desc().Values() , &I_ONE );

	int Num_Lanczos_Steps = 5;
	DblNumMat mat_T(Num_Lanczos_Steps, Num_Lanczos_Steps);
	SetValue(mat_T, 0.0);

	// 0,0 entry is alpha
	mat_T(0,0) = alpha;

	for(Int j = 1; j < Num_Lanczos_Steps; j ++)
	{
		// beta = norm2(f)
		dgdft::scalapack::SCALAPACK(pdnrm2)(&ht, &beta, vec_f.Data(), &I_ONE, &I_ONE, vec_f.Desc().Values(), &I_ONE);
			
		// v0 = v
		dgdft::scalapack::SCALAPACK(pdcopy)(&ht , vec_v.Data() , &I_ONE , &I_ONE , vec_v.Desc().Values() , &I_ONE , 
										vec_v0.Data() , &I_ONE , &I_ONE , vec_v0.Desc().Values() , &I_ONE );

		// v = f / beta
		dgdft::scalapack::SCALAPACK(pdcopy)(&ht , vec_f.Data() , &I_ONE , &I_ONE , vec_f.Desc().Values() , &I_ONE , 
										vec_v.Data() , &I_ONE , &I_ONE , vec_v.Desc().Values() , &I_ONE ); // v = f

		scalar_a = (1.0 / beta);
		dgdft::scalapack::SCALAPACK(pdscal)(&ht , &scalar_a , vec_v.Data() , &I_ONE , &I_ONE , vec_v.Desc().Values(), &I_ONE); // v <-- v (=f) / beta

		// f = H * v : use -H here 
		dgdft::scalapack::SCALAPACK(pdgemv)(&uplo , &ht , &ht , &scalar_minus_one , Hmat.Data() , &I_ONE , &I_ONE , Hmat.Desc().Values() , 
							vec_v.Data() , &I_ONE , &I_ONE , vec_v.Desc().Values() , &I_ONE , 
							&scalar_zero , vec_f.Data(), &I_ONE , &I_ONE , vec_f.Desc().Values() , &I_ONE);

		// f = f - beta * v0
			minus_beta = -beta;
		dgdft::scalapack::SCALAPACK(pdaxpy)(&ht, &minus_beta , vec_v0.Data(), &I_ONE , &I_ONE , vec_v0.Desc().Values() , &I_ONE , 
										vec_f.Data() , &I_ONE , &I_ONE , vec_f.Desc().Values() , &I_ONE );

		// alpha = dot(f,v)
		dgdft::scalapack::SCALAPACK(pddot)(&ht , &alpha ,  vec_f.Data(), &I_ONE , &I_ONE , vec_f.Desc().Values() , &I_ONE , 
										vec_v.Data() ,  &I_ONE , &I_ONE , vec_v.Desc().Values() , &I_ONE );

		// f = f - alpha * v;
			minus_alpha = -alpha;
		dgdft::scalapack::SCALAPACK(pdaxpy)(&ht, &minus_alpha , vec_v.Data(), &I_ONE , &I_ONE , vec_v.Desc().Values() , &I_ONE , 
										vec_f.Data() , &I_ONE , &I_ONE , vec_f.Desc().Values() , &I_ONE );

		// Set up matrix entries
		mat_T(j, j - 1) = beta;
		mat_T(j - 1, j) = beta;
		mat_T(j, j) = alpha;
	} // End of loop over Lanczos steps 

	DblNumVec ritz_values(Num_Lanczos_Steps);
	SetValue( ritz_values, 0.0 );

	// Solve the eigenvalue problem for the Ritz values
	lapack::Syevd( 'N', 'U', Num_Lanczos_Steps, mat_T.Data(), Num_Lanczos_Steps, ritz_values.Data() );

	// Compute the norm of f
	dgdft::scalapack::SCALAPACK(pdnrm2)(&ht, &nrm, vec_f.Data(), &I_ONE, &I_ONE, vec_f.Desc().Values(), &I_ONE);

	// Finally compute upper bound
	b_up = ritz_values(Num_Lanczos_Steps - 1) + nrm;
		
    return b_up;		
}

// This should be called only by prcessors sharing the context of Hmat
void  CheFSI_Hmat_top_parallel(dgdft::scalapack::ScaLAPACKMatrix<Real>& Hmat,
					dgdft::scalapack::ScaLAPACKMatrix<Real>& Xmat,
					DblNumVec& eig_vals_Xmat,
					int filter_order,
					int num_cycles,
					double lower_bound, double upper_bound, double a_L)
{		
	const int ht = Hmat.Height();     
	const int wd = Xmat.Width();
	const int context = Hmat.Context();
	const int scaBlockSize = Hmat.MB();
	
	const int loc_sz = Xmat.LocalHeight() * Xmat.LocalWidth(); 

	double time_sta, time_end;

	double a = lower_bound;
	double b = upper_bound;
	
	// Set up Ymat
	dgdft::scalapack::Descriptor Ymat_desc;
	Ymat_desc.Init( ht, wd, 
			scaBlockSize, scaBlockSize, 
			0, 0, 
			context);   
	dgdft::scalapack::ScaLAPACKMatrix<Real>  Ymat;
	Ymat.SetDescriptor(Ymat_desc);

	// Set up Yt_mat
	dgdft::scalapack::Descriptor Yt_mat_desc;
	Yt_mat_desc.Init( ht, wd, 
			scaBlockSize, scaBlockSize, 
			0, 0, 
			context);   
	dgdft::scalapack::ScaLAPACKMatrix<Real>  Yt_mat;
	Yt_mat.SetDescriptor(Yt_mat_desc);

    statusOFS << std::endl << "  ------- " << std::endl;

	for(int cycle_iter = 1; cycle_iter <= num_cycles; cycle_iter ++)
	{ 
		GetTime(time_sta);
		statusOFS << std::endl << " Inner parallel CheFSI cycle (using -H) iter no. " << cycle_iter << " of " << num_cycles ;
		statusOFS << std::endl << "   Filter order = " << filter_order ;
		statusOFS << std::endl << "   a = " << a << " b = " << b << " a_L = " << a_L ;


		double e = (b - a) / 2.0;
		double c = (a + b) / 2.0;
		double sigma = e / (c - a_L);
		double tau = 2.0 / sigma;

		double sigma_new;

		// A) Compute the filtered subspace
		// Step 1: Y = (H * X - c * X) * (sigma/e)

		// Compute Y = H * X : use -H here
		dgdft::scalapack::Gemm('N', 'N',
									ht, wd, ht,
									-1.0,
									Hmat.Data(), I_ONE, I_ONE, Hmat.Desc().Values(), 
									Xmat.Data(), I_ONE, I_ONE, Xmat.Desc().Values(),
									0.0,
									Ymat.Data(), I_ONE, I_ONE, Ymat.Desc().Values(),
									context);

		// Compute Y = Y - c * X : Use local operations for this
		blas::Axpy( loc_sz, (-c), Xmat.Data(), 1, Ymat.Data(), 1 );

		// Compute Y = Y * (sigma / e) : Use local operations for this
		blas::Scal( loc_sz, (sigma / e), Ymat.Data(), 1 );

		// Loop over filter order

		for(int i = 2; i <= filter_order; i ++)
		{
			sigma_new = 1.0 / (tau - sigma);
		
			// Step 2: Yt = (H * Y - c * Y) * (2 * sigma_new/e) - (sigma * sigma_new) * X

			// Compute Yt = H * Y : use -H here
			dgdft::scalapack::Gemm('N', 'N',
										ht, wd, ht,
										-1.0,
										Hmat.Data(), I_ONE, I_ONE, Hmat.Desc().Values(), 
										Ymat.Data(), I_ONE, I_ONE, Ymat.Desc().Values(),
										0.0,
										Yt_mat.Data(), I_ONE, I_ONE, Yt_mat.Desc().Values(),
										context);
			
			// Compute Yt = Yt - c * Y : Use local operations for this
			blas::Axpy( loc_sz, (-c), Ymat.Data(), 1, Yt_mat.Data(), 1 );

			// Compute Yt = Yt * (2 * sigma_new / e) : Use local operations for this
			blas::Scal( loc_sz, (2.0 * sigma_new / e), Yt_mat.Data(), 1 );

			// Compute Yt = Yt - (sigma * sigma_new) * X : Use local operations for this
			blas::Axpy( loc_sz, (-sigma * sigma_new), Xmat.Data(), 1, Yt_mat.Data(), 1 );

			// Step 3: Update assignments
			
			// Set X = Y : Use local operations for this
			blas::Copy( loc_sz, Ymat.Data(), 1, Xmat.Data(), 1);

			// Set Y = Yt : Use local operations for this
			blas::Copy( loc_sz, Yt_mat.Data(), 1, Ymat.Data(), 1);

			// Set sigma = sigma_new
			sigma = sigma_new;	   
		}
		
		// B) Orthonormalize the filtered vectors
		dgdft::scalapack::Descriptor square_mat_desc;
			square_mat_desc.Init( wd, wd, 
							scaBlockSize, scaBlockSize, 
							0, 0, 
							context);   

		dgdft::scalapack::ScaLAPACKMatrix<Real> square_mat;
		square_mat.SetDescriptor(square_mat_desc);

		// Compute X^T * X
		dgdft::scalapack::Syrk('U', 'T',
						wd, ht,
						1.0, Xmat.Data(),
						I_ONE, I_ONE, Xmat.Desc().Values(),
						0.0, square_mat.Data(),
						I_ONE, I_ONE, square_mat.Desc().Values());

		// Compute the Cholesky factor 
		dgdft::scalapack::Potrf( 'U', square_mat);
			
		// Solve using the Cholesky factor
		// X = X * U^{-1} is orthogonal, where U is the Cholesky factor
		dgdft::scalapack::Trsm('R', 'U', 'N', 'N', 1.0,
									square_mat, 
									Xmat);

		// C) Raleigh-Ritz step
		// Compute Y = H * X : use -H here
		dgdft::scalapack::Gemm('N', 'N',
									ht, wd, ht,
									-1.0,
									Hmat.Data(), I_ONE, I_ONE, Hmat.Desc().Values(), 
									Xmat.Data(), I_ONE, I_ONE, Xmat.Desc().Values(),
									0.0,
									Ymat.Data(), I_ONE, I_ONE, Ymat.Desc().Values(),
									context);

		// Compute X^T * HX
			dgdft::scalapack::Gemm( 'T', 'N',
										wd, wd, ht,
										1.0,
										Xmat.Data(), I_ONE, I_ONE,  Xmat.Desc().Values(), 
										Ymat.Data(), I_ONE, I_ONE,  Ymat.Desc().Values(),
										0.0,
										square_mat.Data(), I_ONE, I_ONE, square_mat.Desc().Values(),
										context);

		// Solve the eigenvalue problem
		std::vector<Real> temp_eigen_values_vector; 	
		dgdft::scalapack::ScaLAPACKMatrix<Real>  scaZ;

		dgdft::scalapack::Syevd('U', square_mat, temp_eigen_values_vector, scaZ);

		// D) Subspace rotation step
		// Copy X to Y
		blas::Copy( loc_sz, Xmat.Data(), 1, Ymat.Data(), 1 );

			// X = X * Q (  Here Ymat contains X)
		dgdft::scalapack::Gemm( 'N', 'N',
									ht, wd, wd,
									1.0,
									Ymat.Data(), I_ONE, I_ONE, Ymat.Desc().Values(), 
									scaZ.Data(), I_ONE, I_ONE, scaZ.Desc().Values(),
									0.0,
									Xmat.Data(), I_ONE, I_ONE,  Xmat.Desc().Values(),
									context);

		// Adjust the lower filter bound for the next cycle 
		a = temp_eigen_values_vector[wd - 1];

		// Flip the sign of the eigenvalues
		for(int iter = 0; iter < wd; iter ++)
			eig_vals_Xmat(iter) = -temp_eigen_values_vector[iter];

		//statusOFS << std::endl << "  CheFSI Eigenvalues in this cycle = " << eig_vals_Xmat << std::endl;

		GetTime(time_end);
		statusOFS << std::endl << " This inner CheFSI cycle finished in " << (time_end - time_sta) << " s." << std::endl;
	}
		
	statusOFS << std::endl << "  ------- " << std::endl;
		
	return;
}

double find_comp_subspace_UB_serial(DblNumMat& Hmat)
{
	double b_up = 0.0;
	int ht = Hmat.m();

	double alpha, beta;

	// Set up a vector v0
	DblNumVec vec_v0(ht);

	// Set up a random vector
	DblNumVec vec_v(ht);
	UniformRandom(vec_v);

	// Normalize this vector
	double vec_v_nrm = blas::Nrm2( ht, vec_v.Data(), 1 );
	blas::Scal( ht, (1.0 / vec_v_nrm),  vec_v.Data(), 1 );

	// Compute f = H * V : use -H here 
	DblNumVec vec_f(ht);
	blas::Gemv( 'N', ht, ht,
		-1.0, Hmat.Data(), ht, 
		vec_v.Data(), 1,
		0.0, vec_f.Data(), 1 );

	// alpha = dot(f,v)
	alpha = blas::Dot( ht, vec_f.Data(), 1, vec_v.Data(), 1 );

	// f = f - alpha * v;
	blas::Axpy( ht, (-alpha), vec_v.Data(), 1, vec_f.Data(), 1 );

	int Num_Lanczos_Steps = 5;
	DblNumMat mat_T(Num_Lanczos_Steps, Num_Lanczos_Steps);
	SetValue(mat_T, 0.0);

	// 0,0 entry is alpha
	mat_T(0,0) = alpha;

	for(Int j = 1; j < Num_Lanczos_Steps; j ++)
		{
	// beta = norm2(f)
	beta = blas::Nrm2(ht, vec_f.Data(), 1);

	// v0 = v
	blas::Copy( ht, vec_v.Data(), 1, vec_v0.Data(), 1 );

	// v = f / beta
	blas::Copy( ht, vec_f.Data(), 1, vec_v.Data(), 1 ); //  v = f
	blas::Scal( ht, (1.0 / beta),  vec_v.Data(), 1 ); // v = v (=f) / beta

	// f = H * v : use -H here 
	blas::Gemv( 'N', ht, ht,
			-1.0, Hmat.Data(), ht, 
			vec_v.Data(), 1,
			0.0, vec_f.Data(), 1 );

	// f = f - beta * v0
	blas::Axpy( ht, (-beta), vec_v0.Data(), 1, vec_f.Data(), 1 );

	// alpha = dot(f,v)
	alpha = blas::Dot( ht, vec_f.Data(), 1, vec_v.Data(), 1 );

	// f = f - alpha * v;
	blas::Axpy( ht, (-alpha), vec_v.Data(), 1, vec_f.Data(), 1 );

	// Set up matrix entries
	mat_T(j, j - 1) = beta;
	mat_T(j - 1, j) = beta;
	mat_T(j, j) = alpha;

		} // End of loop over Lanczos steps 

	DblNumVec ritz_values(Num_Lanczos_Steps);
	SetValue( ritz_values, 0.0 );


	// Solve the eigenvalue problem for the Ritz values
	lapack::Syevd( 'N', 'U', Num_Lanczos_Steps, mat_T.Data(), Num_Lanczos_Steps, ritz_values.Data() );

	// Compute the upper bound
	b_up = ritz_values(Num_Lanczos_Steps - 1) + blas::Nrm2(ht, vec_f.Data(), 1);


	//  statusOFS << std::endl << " Lanczos ritz values = " << ritz_values;
	//  statusOFS << std::endl << " Lanczos upper bound = " << b_up;
	//  
	//   DblNumVec lapack_eigvals_test(ht);
	//   SetValue (lapack_eigvals_test, 0.0);
	//   DblNumMat mat_H_copy(ht,ht);
	//   
	//   blas::Copy( ht*ht, Hmat.Data(), 1, mat_H_copy.Data(), 1 ); 
	//   blas::Scal( ht*ht, -1.0, mat_H_copy.Data(), 1);
	//   
	//   lapack::Syevd( 'N', 'U', ht, mat_H_copy.Data(), ht, lapack_eigvals_test.Data() );
	//   statusOFS << std::endl << lapack_eigvals_test << std::endl;

	return b_up;
}

void  CheFSI_Hmat_top_serial(DblNumMat& H_mat,
				DblNumMat& X_mat,
				DblNumVec& eig_vals_X_mat,
				int filter_order,
				int num_cycles,
				double lower_bound, double upper_bound, double a_L)
{
	int ht = H_mat.m();
	int wd = X_mat.n();

	double time_sta, time_end;

	double a = lower_bound;
	double b = upper_bound;

	for(int cycle_iter = 1; cycle_iter <= num_cycles; cycle_iter ++)
	{ 
		GetTime(time_sta);
		statusOFS << std::endl << " Inner serial CheFSI cycle (using -H) iter no. " << cycle_iter << " of " << num_cycles ;
		statusOFS << std::endl << "   Filter order = " << filter_order ;
		statusOFS << std::endl << "   a = " << a << " b = " << b << " a_L = " << a_L ;

		double e = (b - a) / 2.0;
		double c = (a + b) / 2.0;
		double sigma = e / (c - a_L);
		double tau = 2.0 / sigma;

		double sigma_new;

		DblNumMat Y_mat(ht, wd);
		SetValue(Y_mat, 0.0);

		DblNumMat Yt_mat(ht, wd);
		SetValue(Yt_mat, 0.0);

		// A) Compute the filtered subspace
		// Step 1: Y = (H * X - c * X) * (sigma/e)

		// Compute Y = H * X : use -H here
		blas::Gemm('N', 'N', ht, wd, ht, 
				-1.0, H_mat.Data(), ht,
					X_mat.Data(), ht,
				0.0, Y_mat.Data(), ht);

		// Compute Y = Y - c * X
		blas::Axpy( ht * wd, (-c), X_mat.Data(), 1, Y_mat.Data(), 1 );

		// Compute Y = Y * (sigma / e)
		blas::Scal( ht * wd, (sigma / e), Y_mat.Data(), 1 );

		// Loop over filter order
		for(int i = 2; i <= filter_order; i ++)
			{
			sigma_new = 1.0 / (tau - sigma);

			// Step 2: Yt = (H * Y - c * Y) * (2 * sigma_new/e) - (sigma * sigma_new) * X

			// Compute Yt = H * Y : use -H here
			blas::Gemm('N', 'N', ht, wd, ht, 
					-1.0, H_mat.Data(), ht,
					Y_mat.Data(), ht,
					0.0, Yt_mat.Data(), ht);

			// Compute Yt = Yt - c * Y
			blas::Axpy( ht * wd, (-c), Y_mat.Data(), 1, Yt_mat.Data(), 1 );

			// Compute Yt = Yt * (2 * sigma_new / e)
			blas::Scal( ht * wd, (2.0 * sigma_new / e), Yt_mat.Data(), 1 );

			// Compute Yt = Yt - (sigma * sigma_new) * X
			blas::Axpy( ht * wd, (-sigma * sigma_new), X_mat.Data(), 1, Yt_mat.Data(), 1 );

			// Step 3: Update assignments

			// Set X = Y
			blas::Copy( ht * wd, Y_mat.Data(), 1, X_mat.Data(), 1);

			// Set Y = Yt
			blas::Copy( ht * wd, Yt_mat.Data(), 1, Y_mat.Data(), 1);

			// Set sigma = sigma_new
			sigma = sigma_new;
			}

		// B) Orthonormalize the filtered vectors
		DblNumMat square_mat(wd, wd);

		// Compute X^T * X
		blas::Gemm('T', 'N', wd, wd, ht, 
				1.0, X_mat.Data(), ht,
				X_mat.Data(), ht,
				0.0, square_mat.Data(), wd);

		// Compute the Cholesky factor 
		lapack::Potrf( 'U', wd, square_mat.Data(), wd );

		// Solve using the Cholesky factor
		// X = X * U^{-1} is orthogonal, where U is the Cholesky factor
		blas::Trsm( 'R', 'U', 'N', 'N', 
				ht, wd, 1.0, square_mat.Data(), wd, 
				X_mat.Data(), ht );

		// C) Raleigh-Ritz step

		// Compute Y = H * X : use -H here
		blas::Gemm('N', 'N', ht, wd, ht, 
				-1.0, H_mat.Data(), ht,
					X_mat.Data(), ht,
				0.0, Y_mat.Data(), ht);

		// Compute X^T * HX
		blas::Gemm('T', 'N', wd, wd, ht, 
				1.0, X_mat.Data(), ht,
				Y_mat.Data(), ht,
				0.0, square_mat.Data(), wd);

		// Solve the eigenvalue problem
		eig_vals_X_mat.Resize(wd);
		SetValue(eig_vals_X_mat, 0.0);

		lapack::Syevd( 'V', 'U', wd, square_mat.Data(), wd, eig_vals_X_mat.Data() );

		// D) Subspace rotation step
		// Copy X to Y
		blas::Copy( ht * wd, X_mat.Data(), 1, Y_mat.Data(), 1 );

		// X = X * Q
		blas::Gemm('N', 'N', ht, wd, wd, 
				1.0, Y_mat.Data(), ht,
					square_mat.Data(), wd,
				0.0, X_mat.Data(), ht);

		// Adjust the lower filter bound for the next cycle 
		a = eig_vals_X_mat(wd - 1);

		// Flip the sign of the eigenvalues
		for(int iter = 0; iter < wd; iter ++)
			eig_vals_X_mat(iter) = -eig_vals_X_mat(iter);

		// statusOFS << std::endl << "CheFSI Eigenvalues in this cycle = " << eig_vals_X_mat << std::endl;

		GetTime(time_end);
		statusOFS << std::endl << " This inner CheFSI cycle finished in " << (time_end - time_sta) << " s.";
	}		

	//           DblNumVec lapack_eigvals_test(ht);
	//           SetValue (lapack_eigvals_test, 0.0);
	//           DblNumMat mat_H_copy(ht,ht);
	//       
	//           blas::Copy( ht*ht, H_mat.Data(), 1, mat_H_copy.Data(), 1 ); 
	//           // blas::Scal( ht*ht, -1.0, mat_H_copy.Data(), 1);
	//       
	//           lapack::Syevd( 'N', 'U', ht, mat_H_copy.Data(), ht, lapack_eigvals_test.Data() );
	//           statusOFS << std::endl << " LAPACK Eigenvalues = " << lapack_eigvals_test << std::endl;

	statusOFS << std::endl << "  ---- " << std::endl;

	return;
}

void  LOBPCG_Hmat_top_serial(DblNumMat& Hmat,
				DblNumMat& Xmat,
				DblNumVec& eig_vals_Xmat,
				Int eigMaxIter, Real eigTolerance)
{
	// Do some basic "sanity tests" here
	if( (Hmat.m() != Hmat.n()) ||
	(Hmat.m() != Xmat.m()) ||
	(Xmat.m() <= Xmat.n())
	)
	{
		statusOFS << std::endl << " Error !! Hmat and Xmat are not dimensioned properly !";
		statusOFS << std::endl << " Hmat.m = " << Hmat.m() << " Hmat.n = " << Hmat.n();
		statusOFS << std::endl << " Xmat.m = " << Xmat.m() << " Xmat.n = " << Xmat.n();

		abort();
	}
	statusOFS << std::endl << std::endl;
	// *********************************************************************
	// Initialization
	// *********************************************************************

	Int ntot = Hmat.m();
	Int ncom = 1; // Fixed to be 1 here
	Int noccLocal = Xmat.n(); // Same values in serial version
	Int noccTotal = Xmat.n(); // Serial version

	Int height = ntot * ncom;
	Int widthTotal = noccTotal;
	Int widthLocal = noccLocal;
	Int width = noccTotal;
	Int numEig = Xmat.n();
	Int lda = 3 * width;

	Real timeSta, timeEnd;
	Real timeGemmT = 0.0;
	Real timeGemmN = 0.0;
	Real timeTrsm = 0.0;
	Real timeSpinor = 0.0;
	Real timeMpirank0 = 0.0;
	Int  iterGemmT = 0;
	Int  iterGemmN = 0;
	Int  iterTrsm = 0;
	Int  iterSpinor = 0;
	Int  iterMpirank0 = 0;

	// S = ( X | W | P ) is a triplet used for LOBPCG.  
	// W is the preconditioned residual
	DblNumMat  S( height, 3*widthLocal ), AS( height, 3*widthLocal ); 
	// AMat = S' * (AS),  BMat = S' * S
	// 
	// AMat = (X'*AX   X'*AW   X'*AP)
	//      = (  *     W'*AW   W'*AP)
	//      = (  *       *     P'*AP)
	//
	// BMat = (X'*X   X'*W   X'*P)
	//      = (  *    W'*W   W'*P)
	//      = (  *      *    P'*P)
	//
	DblNumMat  AMat( 3*width, 3*width ), BMat( 3*width, 3*width );
	DblNumMat  AMatT1( 3*width, 3*width );
	// AMatSave and BMatSave are used for restart
	// Temporary buffer array.
	// The unpreconditioned residual will also be saved in Xtemp
	DblNumMat  XTX( width, width ), Xtemp( height, widthLocal );

	// rexNorm Grobal matrix  similar to numEig 
	DblNumVec  resNormLocal ( widthLocal ); 
	SetValue( resNormLocal, 0.0 );
	DblNumVec  resNorm( width );
	SetValue( resNorm, 0.0 );
	Real       resMax, resMin;

	// For convenience
	DblNumMat  X( height, widthLocal, false, S.VecData(0) );
	DblNumMat  W( height, widthLocal, false, S.VecData(widthLocal) );
	DblNumMat  P( height, widthLocal, false, S.VecData(2*widthLocal) );
	DblNumMat AX( height, widthLocal, false, AS.VecData(0) );
	DblNumMat AW( height, widthLocal, false, AS.VecData(widthLocal) );
	DblNumMat AP( height, widthLocal, false, AS.VecData(2*widthLocal) );

	//Int info;
	bool isRestart = false;
	// numSet = 2    : Steepest descent (Davidson), only use (X | W)
	//        = 3    : Conjugate gradient, use all the triplet (X | W | P)
	Int numSet = 2;

	// numLocked is the number of converged vectors
	Int numLockedLocal = 0, numLockedSaveLocal = 0;
	Int numLockedTotal = 0, numLockedSaveTotal = 0; 
	Int numLockedSave = 0;
	// numActive = width - numLocked
	Int numActiveLocal = 0;
	Int numActiveTotal = 0;

	// Real lockTolerance = std::min( eigTolerance_, 1e-2 );

	const Int numLocked = 0;  // Never perform locking in this version
	const Int numActive = width;

	bool isConverged = false;

	// Initialization
	SetValue( S, 0.0 );
	SetValue( AS, 0.0 );

	DblNumVec  eigValS(lda);
	SetValue( eigValS, 0.0 );
	DblNumVec  sigma2(lda);
	DblNumVec  invsigma(lda);
	SetValue( sigma2, 0.0 );
	SetValue( invsigma, 0.0 );

	// Initialize X by the data in Xmat
	lapack::Lacpy( 'A', height, width, Xmat.Data(), height, 
			X.Data(), height );

	// *********************************************************************
	// Main loop
	// *********************************************************************

	// Orthogonalization through Cholesky factorization
	blas::Gemm( 'T', 'N', width, width, height, 1.0, X.Data(), 
		height, X.Data(), height, 0.0, XTX.Data(), width );

	// X'*X=U'*U
	lapack::Potrf( 'U', width, XTX.Data(), width );

	// X <- X * U^{-1} is orthogonal
	blas::Trsm( 'R', 'U', 'N', 'N', height, width, 1.0, XTX.Data(), width, 
		X.Data(), height );

	// Applying the negative of the Hamiltonian matrix : AX = -A * X
	{
		blas::Gemm( 'N', 'N', Hmat.m(), Xmat.n(), Hmat.n(), 
			-1.0, 
			Hmat.Data(), Hmat.m(),
			Xmat.Data(), Xmat.m(), 
			0.0, AX.Data(), AX.m() );
	}

	// Start the main loop
	Int iter;
	for( iter = 1; iter < eigMaxIter; iter++ )
	{
		if( iter == 1 || isRestart == true )
			numSet = 2;
		else
			numSet = 3;

		SetValue( AMat, 0.0 );
		SetValue( BMat, 0.0 );

		// Rayleigh Ritz in Q = span( X, gradient )

		// XTX <- X' * (AX)
		blas::Gemm( 'T', 'N', width, width, height, 1.0, X.Data(),
				height, AX.Data(), height, 0.0, XTX.Data(), width );

		lapack::Lacpy( 'A', width, width, XTX.Data(), width, AMat.Data(), lda );

		// Compute the residual.
		// R <- AX - X*(X'*AX)
		lapack::Lacpy( 'A', height, width, AX.Data(), height, Xtemp.Data(), height );

		blas::Gemm( 'N', 'N', height, width, width, -1.0, 
				X.Data(), height, AMat.Data(), lda, 1.0, Xtemp.Data(), height );

		// Compute the norm of the residual
		SetValue( resNorm, 0.0 );
	    for( Int k = 0; k < widthLocal; k++ )
		{
		    resNorm( k ) = std::sqrt( Energy(DblNumVec(height, false, Xtemp.VecData(k)))); 
		}

		resMax = *(std::max_element( resNorm.Data(), resNorm.Data() + numEig ) );
		resMin = *(std::min_element( resNorm.Data(), resNorm.Data() + numEig ) );

		statusOFS << " Serial LOBPCG iter = " << iter << " maxRes = " << resMax << std::endl;

		if( resMax < eigTolerance ){
			isConverged = true;;
			break;
		}

		//numActive = width - numLocked;
		numActiveLocal = widthLocal - numLockedLocal;
		numActiveTotal = widthTotal - numLockedTotal;

		// If the number of locked vectors goes down, perform steepest
		// descent rather than conjugate gradient
		// if( numLockedTotal < numLockedSaveTotal )
		//  numSet = 2;

		// Compute the preconditioned residual W = T*R.
		// Here, no preconditioner is applied
		{
			lapack::Lacpy( 'A', height, width, Xtemp.Data(), height, W.Data(), height );
		}

		//    Real normLocal; 
		Real norm; 

		norm = 0.0; 
		// Normalize the preconditioned residual
		for( Int k = numLockedLocal; k < widthLocal; k++ ){
			norm = Energy(DblNumVec(height, false, W.VecData(k)));
			norm = std::sqrt( norm );
			blas::Scal( height, 1.0 / norm, W.VecData(k), 1 );
		}

		norm = 0.0; 
		// Normalize the conjugate direction
		if( numSet == 3 ){
			for( Int k = numLockedLocal; k < widthLocal; k++ ){
			norm = Energy(DblNumVec(height, false, P.VecData(k)));
			norm = std::sqrt( norm );
			blas::Scal( height, 1.0 / norm, P.VecData(k), 1 );
			blas::Scal( height, 1.0 / norm, AP.VecData(k), 1 );
			}
		}

		// Compute AMat

		// Applying the negative of the Hamiltonian matrix : AX = -A * X
		{
			blas::Gemm( 'N', 'N', Hmat.m(), W.n(), Hmat.n(), 
					-1.0, 
					Hmat.Data(), Hmat.m(),
					W.Data(), W.m(), 
					0.0, AW.Data(), AW.m() );
		}

		// Compute X' * (AW)
		// Instead of saving the block at &AMat(0,width+numLocked), the data
		// is saved at &AMat(0,width) to guarantee a continuous data
		// arrangement of AMat.  The same treatment applies to the blocks
		// below in both AMat and BMat.
		blas::Gemm( 'T', 'N', width, numActive, height, 1.0, X.Data(),
				height, AW.VecData(numLocked), height, 
				0.0, &AMat(0,width), lda );

		// Compute W' * (AW)
		blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
				W.VecData(numLocked), height, AW.VecData(numLocked), height, 
				0.0, &AMat(width, width), lda );

		if( numSet == 3 ){
			// Compute X' * (AP)
			blas::Gemm( 'T', 'N', width, numActive, height, 1.0,
					X.Data(), height, AP.VecData(numLocked), height, 
					0.0, &AMat(0, width+numActive), lda );

			// Compute W' * (AP)
			blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
					W.VecData(numLocked), height, AP.VecData(numLocked), height, 
					0.0, &AMat(width, width+numActive), lda );

			// Compute P' * (AP)
			blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
					P.VecData(numLocked), height, AP.VecData(numLocked), height, 
					0.0, &AMat(width+numActive, width+numActive), lda );
		}

		// Compute BMat (overlap matrix)

		// Compute X'*X
		blas::Gemm( 'T', 'N', width, width, height, 1.0, 
				X.Data(), height, X.Data(), height, 
				0.0, &BMat(0,0), lda );

		// Compute X'*W
		blas::Gemm( 'T', 'N', width, numActive, height, 1.0,
				X.Data(), height, W.VecData(numLocked), height,
				0.0, &BMat(0,width), lda );

		// Compute W'*W
		blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
				W.VecData(numLocked), height, W.VecData(numLocked), height,
				0.0, &BMat(width, width), lda );

		if( numSet == 3 ){
			// Compute X'*P
			blas::Gemm( 'T', 'N', width, numActive, height, 1.0,
					X.Data(), height, P.VecData(numLocked), height, 
					0.0, &BMat(0, width+numActive), lda );

			// Compute W'*P
			blas::Gemm('T', 'N', numActive, numActive, height, 1.0,
					W.VecData(numLocked), height, P.VecData(numLocked), height,
					0.0, &BMat(width, width+numActive), lda );

			// Compute P'*P
			blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
					P.VecData(numLocked), height, P.VecData(numLocked), height,
					0.0, &BMat(width+numActive, width+numActive), lda );
		}

		#if ( _DEBUGlevel_ >= 2 )
		{
			DblNumMat WTW( width, width );
			lapack::Lacpy( 'A', width, width, &BMat(width, width), lda,
					WTW.Data(), width );


			statusOFS << "W'*W = " << WTW << std::endl;
			if( numSet == 3 )
			{
				DblNumMat PTP( width, width );
				lapack::Lacpy( 'A', width, width, &BMat(width+numActive, width+numActive), 
						lda, PTP.Data(), width );

				statusOFS << "P'*P = " << PTP << std::endl;
			}
		}
		#endif

		// Rayleigh-Ritz procedure
		// AMat * C = BMat * C * Lambda
		// Assuming the dimension (needed) for C is width * width, then
		//     ( C_X )
		//     ( --- )
		// C = ( C_W )
		//     ( --- )
		//     ( C_P )
		//
		Int numCol;
		if( numSet == 3 ){
			// Conjugate gradient
			numCol = width + 2 * numActiveTotal;
		}
		else{
			numCol = width + numActiveTotal;
		}

		// Solve the generalized eigenvalue problem with thresholding

		// Symmetrize A and B first.  This is important.
		for( Int j = 0; j < numCol; j++ ){
			for( Int i = j+1; i < numCol; i++ ){
			AMat(i,j) = AMat(j,i);
			BMat(i,j) = BMat(j,i);
			}
		}

		lapack::Syevd( 'V', 'U', numCol, BMat.Data(), lda, sigma2.Data() );

		Int numKeep = 0;
		for( Int i = numCol-1; i>=0; i-- ){
			if( sigma2(i) / sigma2(numCol-1) >  1e-12 )
			numKeep++;
			else
			break;
		}

		#if ( _DEBUGlevel_ >= 1 )
		statusOFS << "sigma2 = " << sigma2 << std::endl;
		#endif

		#if ( _DEBUGlevel_ >= 1 )
		statusOFS << "sigma2(0)        = " << sigma2(0) << std::endl;
		statusOFS << "sigma2(numCol-1) = " << sigma2(numCol-1) << std::endl;
		statusOFS << "numKeep          = " << numKeep << std::endl;
		#endif

		for( Int i = 0; i < numKeep; i++ ){
			invsigma(i) = 1.0 / std::sqrt( sigma2(i+numCol-numKeep) );
		}

		if( numKeep < width ){
			std::ostringstream msg;
			msg 
			<< "width   = " << width << std::endl
			<< "numKeep =  " << numKeep << std::endl
			<< "there are not enough number of columns." << std::endl;
			ErrorHandling( msg.str().c_str() );
		}

		SetValue( AMatT1, 0.0 );
		// Evaluate S^{-1/2} (U^T A U) S^{-1/2}
		blas::Gemm( 'N', 'N', numCol, numKeep, numCol, 1.0,
				AMat.Data(), lda, BMat.VecData(numCol-numKeep), lda,
				0.0, AMatT1.Data(), lda );

		blas::Gemm( 'T', 'N', numKeep, numKeep, numCol, 1.0,
				BMat.VecData(numCol-numKeep), lda, AMatT1.Data(), lda, 
				0.0, AMat.Data(), lda );

		for( Int j = 0; j < numKeep; j++ ){
			for( Int i = 0; i < numKeep; i++ ){
			AMat(i,j) *= invsigma(i)*invsigma(j);
			}
		}

		// Solve the standard eigenvalue problem
		lapack::Syevd( 'V', 'U', numKeep, AMat.Data(), lda,
					eigValS.Data() );

		// Compute the correct eigenvectors and save them in AMat
		for( Int j = 0; j < numKeep; j++ ){
			for( Int i = 0; i < numKeep; i++ ){
			AMat(i,j) *= invsigma(i);
			}
		}

		blas::Gemm( 'N', 'N', numCol, numKeep, numKeep, 1.0,
				BMat.VecData(numCol-numKeep), lda, AMat.Data(), lda,
				0.0, AMatT1.Data(), lda );

		lapack::Lacpy( 'A', numCol, numKeep, AMatT1.Data(), lda, 
					AMat.Data(), lda );

		if( numSet == 2 ){
			// Update the eigenvectors 
			// X <- X * C_X + W * C_W
			blas::Gemm( 'N', 'N', height, width, width, 1.0,
					X.Data(), height, &AMat(0,0), lda,
					0.0, Xtemp.Data(), height );

			blas::Gemm( 'N', 'N', height, width, numActive, 1.0,
					W.VecData(numLocked), height, &AMat(width,0), lda,
					1.0, Xtemp.Data(), height );

			// Save the result into X
			lapack::Lacpy( 'A', height, width, Xtemp.Data(), height, 
					X.Data(), height );

			// P <- W
			lapack::Lacpy( 'A', height, numActive, W.VecData(numLocked), 
					height, P.VecData(numLocked), height );
		}
		else
		{ 
			//numSet == 3
			// Compute the conjugate direction
			// P <- W * C_W + P * C_P
			blas::Gemm( 'N', 'N', height, width, numActive, 1.0,
				W.VecData(numLocked), height, &AMat(width, 0), lda, 
				0.0, Xtemp.Data(), height );



			blas::Gemm( 'N', 'N', height, width, numActive, 1.0,
				P.VecData(numLocked), height, &AMat(width+numActive,0), lda,
				1.0, Xtemp.Data(), height );


			lapack::Lacpy( 'A', height, numActive, Xtemp.VecData(numLocked), 
					height, P.VecData(numLocked), height );

			// Update the eigenvectors
			// X <- X * C_X + P
			blas::Gemm( 'N', 'N', height, width, width, 1.0, 
				X.Data(), height, &AMat(0,0), lda, 
				1.0, Xtemp.Data(), height );


			lapack::Lacpy( 'A', height, width, Xtemp.Data(), height,
					X.Data(), height );
		} // if ( numSet == 2 ) else...

		// Update AX and AP
		if( numSet == 2 ){
			// AX <- AX * C_X + AW * C_W
			blas::Gemm( 'N', 'N', height, width, width, 1.0,
					AX.Data(), height, &AMat(0,0), lda,
					0.0, Xtemp.Data(), height );

			blas::Gemm( 'N', 'N', height, width, numActive, 1.0,
					AW.VecData(numLocked), height, &AMat(width,0), lda,
					1.0, Xtemp.Data(), height );

			lapack::Lacpy( 'A', height, width, Xtemp.Data(), height,
					AX.Data(), height );

			// AP <- AW
			lapack::Lacpy( 'A', height, numActive, AW.VecData(numLocked), height,
					AP.VecData(numLocked), height );
		}
		else
		{
		// AP <- AW * C_W + A_P * C_P
		blas::Gemm( 'N', 'N', height, width, numActive, 1.0, 
			AW.VecData(numLocked), height, &AMat(width,0), lda,
			0.0, Xtemp.Data(), height );

		blas::Gemm( 'N', 'N', height, width, numActive, 1.0,
			AP.VecData(numLocked), height, &AMat(width+numActive, 0), lda,
			1.0, Xtemp.Data(), height );

		lapack::Lacpy( 'A', height, numActive, Xtemp.VecData(numLocked), 
				height, AP.VecData(numLocked), height );

		// AX <- AX * C_X + AP
		blas::Gemm( 'N', 'N', height, width, width, 1.0,
			AX.Data(), height, &AMat(0,0), lda,
			1.0, Xtemp.Data(), height );


		lapack::Lacpy( 'A', height, width, Xtemp.Data(), height, 
				AX.Data(), height );
		} // if ( numSet == 2 ) else ...
#if ( _DEBUGlevel_ >= 1 )
		statusOFS << "numLocked = " << numLocked << std::endl;
		statusOFS << "eigValS   = " << eigValS << std::endl;
#endif
	} // for (iter) end for main loop

	// *********************************************************************
	// Post processing
	// *********************************************************************

	// Obtain the eigenvalues and eigenvectors
	// XTX should now contain the matrix X' * (AX), and X is an
	// orthonormal set
	lapack::Syevd( 'V', 'U', width, XTX.Data(), width, eigValS.Data() );

	// X <- X*C
	blas::Gemm( 'N', 'N', height, width, width, 1.0, X.Data(),
		height, XTX.Data(), width, 0.0, Xtemp.Data(), height );

	lapack::Lacpy( 'A', height, width, Xtemp.Data(), height,
			X.Data(), height );

#if ( _DEBUGlevel_ >= 2 )
	blas::Gemm( 'T', 'N', width, width, height, 1.0, X.Data(), 
		height, X.Data(), height, 0.0, XTX.Data(), width );

	statusOFS << "After the LOBPCG, XTX = " << XTX << std::endl;
#endif

	// Copy back results
	lapack::Lacpy( 'A', height, width, X.Data(), height, 
			Xmat.Data(), height );
	for(Int copy_iter = 0; copy_iter < width; copy_iter ++)
		eig_vals_Xmat(copy_iter) = -eigValS(copy_iter); 

	//statusOFS << std::endl << " Eigvals = " << std::endl <<  eigValS;
	//statusOFS << std::endl << " Res = " << std::endl <<  resNorm; 

	if( isConverged ){
		statusOFS << std::endl << " After " << iter 
		<< " iterations, LOBPCG has converged."  << std::endl
		<< " The maximum norm of the residual is " 
		<< resMax << std::endl << std::endl;
	}
	else{
		statusOFS << std::endl << " After " << iter 
		<< " iterations, LOBPCG did not converge. " << std::endl
		<< " The maximum norm of the residual is " 
		<< resMax << std::endl 
		<< " Desired max residual = " << eigTolerance << std::endl
		<< std::endl;
	}

	return ;
}     

}

#endif
