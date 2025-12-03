
#include "global.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/*  out-of-place algorithm, row-major
 *
 *  given X(1:n1,1:n2,1:n3), 
 *  transpose to Y(1:n2,1:n3,1:n1) via (x,y,z) --> (y,z,x) 
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
void transpose3D_cpu( doublereal* Y, doublereal* X, 
    const unsigned int n1, const unsigned int n2, const unsigned int n3 )
{
    unsigned int i, j, k ;
    unsigned int index_in, index_out ;

    for ( i = 1 ; i <= n1 ; i++ ){
        for ( j =1 ; j <= n2 ; j++ ){
            for ( k = 1 ; k <= n3 ; k++){
                // X(1:n1,1:n2,1:n3) tranpose to  Y(1:n2,1:n3,1:n1)
                // Y(j,k,i) = X(i,j,k)
                index_in  = (i-1)*n2*n3 + (j-1)*n3 + (k-1) ;
                index_out = (j-1)*n3*n1 + (k-1)*n1 + (i-1) ;
                Y[index_out] = X[index_in] ;
            }
        }
    }
 
}



