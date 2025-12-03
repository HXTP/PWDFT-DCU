
/*
 *  test transpose 3D, row-major
 *
 */

#include "global.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include <cuda_runtime.h>

/*
 *  Y = tranpose(X) under rule xyz2yzx
 */
void transpose3D_cpu( doublereal* Y, doublereal* X, 
    const unsigned int n1, const unsigned int n2, const unsigned int n3 ) ;

void  transpose3D_revised_xyz2yzx_device( doublereal *Y, doublereal *X, 
    unsigned int n1, unsigned int n2, unsigned int n3 ) ;

typedef struct {
    unsigned int n1 ;
    unsigned int n2 ;
    unsigned int n3 ;
    int  deviceID ;
} optsPara ;

void showHelp(void)
{
    printf("useage: [bin] -n1 [n1] -n2 [n2] -n3 [n3] -G[GPU id]\n");
    printf("3-D matrix is row-major and dimension is n1 x n2 x n3 \n");
    printf("default of [GPU id] is 0 \n");
}

void processCommandOption( int argc, char** argv, optsPara *opts)
{
    if ( 2 > argc ){
        showHelp();
        exit(1) ;
    }
    argc-- ;
    argv++ ;

    for( ; argc > 0 ; argc-- , argv++){
        if ( '-' != argv[0][0] ){ continue ; }
        if ( 'G' == argv[0][1] ){
            opts->deviceID = argv[0][2] - '0';
            continue ;
        }
        if ( 0 == strcmp("-n1", argv[0]) ){
            argc-- ;
            argv++ ;
            if ( 0 >= argc ){
                fprintf(stderr, "Error: miss integer after option -n1\n");
                exit(1) ;
            }
            opts->n1 = atoi( argv[0] );
            continue ;
        }
        if ( 0 == strcmp("-n2", argv[0]) ){
            argc-- ;
            argv++ ;
            if ( 0 >= argc ){
                fprintf(stderr, "Error: miss integer after option -n2\n");
                exit(1) ;
            }
            opts->n2 = atoi( argv[0] );
            continue ;
        }
        if ( 0 == strcmp("-n3", argv[0]) ){
            argc-- ;
            argv++ ;
            if ( 0 >= argc ){
                fprintf(stderr, "Error: miss integer after option -n3\n");
                exit(1) ;
            }
            opts->n3 = atoi( argv[0] );
            continue ;
        }
    }

    if ( (0 == opts->n1) || (0 == opts->n2) || (0 == opts->n3) ){
        showHelp();
        exit(1);
    }

}

int main( int argc, char* argv[] )
{
    cudaError_t status ;

    int numIterations = 100 ; // number of iterations

    doublereal  *X ; // input 
    doublereal  *reference ; // reference = cpu_transpose3D(X) 
    doublereal  *Y ; // Y = gpu_transpose(X) 
    doublereal  *d_X ; // X in device memory
    doublereal  *d_Y ; // Y in device memory 

    optsPara opts ;

    memset( &opts, 0, sizeof(optsPara) ) ;
    processCommandOption( argc, argv, &opts) ;

    const unsigned int n1 = opts.n1 ;
    const unsigned int n2 = opts.n2 ;
    const unsigned int n3 = opts.n3 ;

    cudaDeviceProp deviceProp;
    int device = opts.deviceID ;
    cudaGetDeviceProperties(&deviceProp, device);
    cudaSetDevice( device ) ;
    printf("use device %d, name = %s\n", device, deviceProp.name );

    printf("n1 = %d, n2 = %d, n3 = %d\n", n1, n2, n3 );
    printf("numIterations = %d\n", numIterations );

    X = (doublereal*)malloc( sizeof(doublereal) *n1*n2*n3 );
    assert( X ) ;

    reference = (doublereal*)malloc( sizeof(doublereal) *n1*n2*n3 ) ;
    assert( reference ) ;

    memset( reference, 0, sizeof(doublereal) *n1*n2*n3 );

    Y = (doublereal*)malloc( sizeof(doublereal) *n1*n2*n3 ) ;
    assert( Y ) ;

    // step 1: generate input X, row-major
    for ( int i = 1 ; i <= n1 ; i++ ){
        for ( int j = 1 ; j <= n2 ; j++ ){
            for ( int  k = 1 ; k <= n3 ; k++){
                int index  = (i-1)*n2*n3 + (j-1)*n3 + (k-1) ;
                X[index] = index ;
            }
        }
    }

    // step 2: compute reference = tranpose(X) in CPU 
    transpose3D_cpu( reference, X, n1, n2, n3 );

    // step 3: compute Y = X in GPU
    status = cudaMalloc((void**)&d_X, sizeof(doublereal)*n1*n2*n3 );
    assert( cudaSuccess == status );

    status = cudaMalloc((void**)&d_Y, sizeof(doublereal)*n1*n2*n3 );
    assert( cudaSuccess == status );
    
    status = cudaMemcpy( d_X, X, sizeof(doublereal)*n1*n2*n3, cudaMemcpyHostToDevice);
    assert( cudaSuccess == status );

    status = cudaMemset( d_Y, 0, sizeof(doublereal)*n1*n2*n3 );
    assert( cudaSuccess == status );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int i = 0; i < numIterations; ++i){
        transpose3D_revised_xyz2yzx_device(d_Y, d_X, n1, n2, n3 ); 
    }
    // record time setting 
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float naiveTime ;
    cudaEventElapsedTime(&naiveTime, start, stop);

    // compute d2d bandwidth
    double gpu_time  = naiveTime / ((double)numIterations) ;
    double rw_inByte = sizeof(doublereal) * n1 * n2 * n3 ; 
    rw_inByte *= 2.0 ; // read and write
	
    double bandwidth_GB = (rw_inByte /gpu_time) * ( 1000.0 / 1024.0 ) /1024.0 / 1024.0 ;

    printf("GPU elapsed time:  %0.3f ms\n", gpu_time );
    printf("GPU bandwidth = %0.2f GB/s\n", bandwidth_GB );

    status = cudaMemcpy( Y, d_Y, sizeof(doublereal)*n1*n2*n3, cudaMemcpyDeviceToHost);
    assert( cudaSuccess == status );

    // step 4: veriyf data 
    doublereal max_err = 0.0 ;
    for( unsigned int i = 0 ; i < (n1*n2*n3) ; i++ ){
        doublereal err = fabs(Y[i] - reference[i]);
        if ( max_err < err ){
            max_err = err ;
        }
    }    
    printf("maxError = %E\n", max_err );

    free(X) ;
    free(reference) ;
    free(Y) ;
    cudaFree( d_X );
    cudaFree( d_Y );
 
    cudaThreadExit();

    return 0 ;
}


