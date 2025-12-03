//#include <cuda.h>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdlib.h>

#define NX 4
#define NY 4
int main()
{
        float *vx = (float*) malloc( NX * NY * sizeof(cufftComplex));
        cufftComplex *d_vx, *d_vx2;
     	int i;
    	for(i =0; i < NX *NY ; i++)
    		vx[i] = 1.0;
	assert(cudaSetDevice(0) == hipSuccess);
	printf("NX NY sizoef(cufftComplex): %d %d %d \n", NX, NY, sizeof(cufftComplex));
        assert(cudaMalloc(&d_vx,  NX*NY*sizeof(cufftComplex)) == hipSuccess);
        assert(cudaMalloc(&d_vx2, NX*NY*sizeof(cufftComplex)) == hipSuccess);
        assert(cudaMemcpy(d_vx, vx, NX*NY*sizeof(cufftComplex), cudaMemcpyHostToDevice) == hipSuccess);
    	for(i =0; i < NX *NY ; i++)
    		vx[i] = 0.0;
        cufftHandle planr2c;
        cufftHandle planc2r;
        (cufftPlan2d(&planr2c, NY, NX, CUFFT_R2C));
        (cufftPlan2d(&planc2r, NY, NX, CUFFT_C2R));
        //(cufftSetCompatibilityMode(planr2c, CUFFT_COMPATIBILITY_NATIVE));
        //(cufftSetCompatibilityMode(planc2r, CUFFT_COMPATIBILITY_NATIVE));
        assert(cufftExecR2C(planr2c, (cufftReal *)d_vx, d_vx2) == CUFFT_SUCCESS);
        //cufftExecC2R(planc2r, d_vx, (cufftReal *)d_vx2);
        assert(cudaMemcpy(vx, d_vx2, NX*NY*sizeof(cufftComplex), cudaMemcpyDeviceToHost) == hipSuccess);
	for(i =0; i < NX *NY; i++)
		printf(" vx[%d]: %f \n" , i, vx[i]);
        cudaFree(d_vx);
        cudaFree(d_vx2);
        free(vx);
}
