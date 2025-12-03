

/*  File: transpose3D_revised.cu
 *  author: Lung-Sheng Chien
 *  department of Mathematics, national Tsing Hua univeristy, R.O.C (Taiwan)
 *  email: d947207@oz.nthu.edu.tw
 * 
 *  description: transpose 3-D matrix A(i,j,k) --> B(j,k,i)
 */

#include "global.h"

#define BLOCK_DIM 16

#include <stdio.h>
#include <stdlib.h>

__global__ void  transpose3D_revised_xyz2yzx( doublereal *odata, doublereal *idata, 
    unsigned int n1, unsigned int n2, unsigned int n3, 
    unsigned int Gx, unsigned int Gz, 
    float one_over_Gx, float one_over_Gz, unsigned int k2 ) ;
				
/*  out-of-place algorithm
 *
 *  given X(1:n1,1:n2,1:n3), transpose to Y(1:n2,1:n3,1:n1) via (x,y,z) --> (y,z,x) 
 *
 *  CPU code
 *
 *      for i = 1:n1
 *          for j = 1:n2
 *              for k = 1:n3
 *                  Y(j,k,i) = X(i,j,k)
 *              end
 *          end
 *      end
 *
 *
 *  Output:
 *      Y(1:n2,1:n3,1:n1): device memory, Y can not be aliased to X
 *
 *  Input:
 *      X(1:n1,1:n2,1:n3): device memory
 *      n1: number of data point in x-axis
 *      n2: number of data point in y-axis
 *      n3: number of data point in z-axis 
 *		
 */			
void  transpose3D_revised_xyz2yzx_device( doublereal *Y, doublereal *X, 
    unsigned int n1, unsigned int n2, unsigned int n3 ) 
			
{
    unsigned int Gx, Gz, k1, k2 ;
    double db_n2 = (double) n2 ;
	
    // we only accept out-of-place 
    if ( X == Y ){
        printf("Error(transpose3D_xyz2yzx_device): we only accept out-of-place \n");
        exit(1) ;
    }	

   /*  Gx = number of grids need in x-axis
    *  Gz = number of grids need in z-axis
    *
    *  we call a coarse grid is compose of grid Gx x Gz 
    */	
    Gx = (n1 + BLOCK_DIM-1) / BLOCK_DIM ; 
    Gz = (n3 + BLOCK_DIM-1) / BLOCK_DIM ; 
	
   /*
    *  since a coarse can cover a x-z slice, we need n2 corase grids to cover X
    *  
    *  in order to save resource, we want to find two integers k1, k2 such that 
    *
    *       k1 * k2 - n2 <= 1
    *
    *  for example: 
    *       n2 = 7   ==> k1 = 2 and k2 = 4
    *       n2 = 13  ==> k2 = 2 and k2 = 7 
    *
    */	
    int max_k1 = (int) floor( sqrt(db_n2) ) ;
    for ( k1 = max_k1 ; 1 <= k1 ; k1-- ){
        k2 = (unsigned int) ceil( db_n2/((double)k1)) ;
        if ( 1 >= (k1*k2 - n2) ){
            break ;
        }
    }
	
//  printf("n2 = %d, k1 = %d, k2 = %d\n", n2, k1, k2);
	
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 grid( k2*Gz, k1*Gx, 1 );
  
    float eps = 1.E-5 ;
    float one_over_Gx = (1.0 + eps)/((double)Gx) ; 
    float one_over_Gz = (1.0 + eps)/((double)Gz) ;
    
    transpose3D_revised_xyz2yzx<<< grid, threads >>>( Y, X, 
        n1, n2, n3, Gx, Gz, one_over_Gx, one_over_Gz, k2 ) ;
			 
}

/*
 *  given X(1:n1,1:n2,1:n3), transpose to Y(1:n2,1:n3,1:n1) via (x,y,z) --> (y,z,x) 
 *  
 *  Assumption: this is out-of-place algorithm, Y can not be aliased to X
 *      this assertion holds if one calls
 *      transpose3D_xyz2yzx_device( X, X, n1, n2, n3 ) 
 *  
 *  output:
 *      odata = Y(1:n2,1:n3,1:n1)
 *  Input:
 *      idata = X(1:n1,1:n2,1:n3)
 *      n1: number of data point in x-axis
 *      n2: number of data point in y-axis
 *      n3: number of data point in z-axis 
 *      Gx: number of grid in x-axis in x-z slice, Gx = (n1+BLOCK_DIM-1)/BLOCK_DIM 
 *      Gz: number of grid in z-axis in x-z slice, Gz = (n3+BLOCK_DIM-1)/BLOCK_DIM 
 *      k2: number of coarse grid in z-axis, where k1: number of coarse grid in x-axis
 *          grid( k2 * Gx, k1 * Gx)
 *          blocks( BLOCK_DIM, BLOCK_DIM )
 *
 *  utility function 
 *      1. float floorf(float x) 
 *         The floor() and floorf() functions return the largest integral value 
 *         less than or equal to x. 
 *
 *      2. __int2float_rz( int x ) or __uint2float_( uint x)
 *         Functions suffixed with _rz operate using the round-towards-zero rounding mode   
 * 
 *      3. __float2int_( float x ) or __float2uint_( float x )
 *
 *
 *  built-in type:
 *
 *  gridDim, blockDim: dim3
 *  blockIdx, threadIdx: uint3
 *
 *	struct dim3
 *	{
 *  	 unsigned int x, y, z;
 *		#if defined(__cplusplus)
 *   		__host__ __device__ dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) {}
 *   		__host__ __device__ dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
 *   		__host__ __device__ operator uint3(void) { uint3 t; t.x = x; t.y = y; t.z = z; return t; }
 *		#endif 
 *	};
 *
 *	struct uint3
 *	{
 *	  unsigned int x, y, z;
 *	  __cuda_assign_operators(uint3)
 *	}; 
 *
 *
 * This kernel is optimized to ensure all global reads and writes are coalesced,
 * and to avoid bank conflicts in shared memory. 
 * Note that the shared memory array is sized to (BLOCK_DIM+1)*BLOCK_DIM.  
 * This pads each row of the 2D block in shared memory so that bank conflicts do not occur 
 * when threads address the array column-wise.
 *
 */

__global__ void  transpose3D_revised_xyz2yzx( doublereal *odata, doublereal *idata, 
    unsigned int n1, unsigned int n2, unsigned int n3, 
    unsigned int Gx, unsigned int Gz, 
    float one_over_Gx, float one_over_Gz, unsigned int k2 )
{
    __shared__ doublereal block[BLOCK_DIM][BLOCK_DIM+1];

    float tmp1 ;
    unsigned int s1, s2, t1, t2 ;
    unsigned int xIndex, yIndex, zIndex ;
    unsigned int index_in, index_out ;
	
   /* step 1: transform grid index to 3D corase grid index
    *   blockIdx.x = Gz * s1 + t1 
    *   blockIdx.y = Gx * s2 + t2 
    *
    *   where (s1, s2): index to y-direction, (t1, t2): index to x-z slice (local blockID )
    *
    *  s1 = floorf( blockIdx.x / Gz ) 
    *  t1 = blockIdx.x - Gz*s1
    *
    *  s2 = floorf( blockIdx.y / Gx )  
    *  t2 = blockIdx.y - Gx*s2
    */	
    tmp1 = __uint2float_rz( blockIdx.x ) ;
    tmp1 = floorf( tmp1 * one_over_Gz ) ;
    s1 = __float2uint_rz( tmp1 ) ; 
    t1 = blockIdx.x - Gz*s1 ;
 	
    tmp1 = __uint2float_rz( blockIdx.y ) ;
    tmp1 = floorf( tmp1 * one_over_Gx ) ;
    s2 = __float2uint_rz( tmp1 ) ; 
    t2 = blockIdx.y - Gx*s2 ;
 
   /* 
    *  step 2: yIndex = s2*k2 + s1 from (s1, s2)
    */ 
    yIndex = s2*k2 + s1 ;
 
   /*
    * step 3: read the matrix tile into shared memory  
    *
    */
    zIndex = t1 * BLOCK_DIM + threadIdx.x ;
    xIndex = t2 * BLOCK_DIM + threadIdx.y ;

    if ( (yIndex < n2) && (xIndex < n1) && (zIndex < n3)  ){
        index_in = (xIndex * n2 + yIndex) * n3 + zIndex ; 
        block[threadIdx.y][threadIdx.x] = idata[index_in];
    }
    __syncthreads();

   /*
    * step 4: write the transposed matrix tile to global memory
    *
    */
    xIndex = t2 * BLOCK_DIM + threadIdx.x ;
    zIndex = t1 * BLOCK_DIM + threadIdx.y ;
 	
    if ( (yIndex < n2) && (xIndex < n1) && (zIndex < n3)  ){
        index_out = (yIndex * n3 + zIndex) * n1 + xIndex ; 
        odata[index_out] = block[threadIdx.x][threadIdx.y] ;
    } 	
 
}


