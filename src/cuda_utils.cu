//#include "utility.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hip/hip_runtime.h>
#ifndef  _CUDA_UTILS_CU_
#define  _CUDA_UTILS_CU_
#include "cuda_utils.h"
//#include "hip_complex.h"
#define DIM   128
#define LDIM  256
#define LEN   512
#define BLOCK_SIZE 1024
//typedef  hipDoubleComplex cuDoubleComplex ;
//typedef  hipFloatComplex cuComplex;
double * dev_vtot;
double * dev_gkkR2C;
int    * dev_idxFineGridR2C;
int    * dev_NLindex;
int    * dev_NLpart;
double * dev_NLvecFine;
double * dev_atom_weight;
double * dev_temp_weight;
double * dev_TeterPrecond;

bool vtot_gpu_flag;
bool NL_gpu_flag;
bool teter_gpu_flag;
int totPart_gpu;

#define gpuErrchk(ans) {gpuAssert((ans),__FILE__,__LINE__);}
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true)
{
	if(code != hipSuccess){
		fprintf(stderr, "GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
		if(abort) exit(code);
	}
}
/*
__device__ inline cuDoubleComplex operator* (const cuDoubleComplex & x,const cuDoubleComplex & y) {
	return hipCmul(x,y);
}

__device__ inline cuDoubleComplex operator+ (const cuDoubleComplex & x,const cuDoubleComplex & y) {
	return hipCadd(x,y);
}

__device__ inline cuDoubleComplex operator- (const cuDoubleComplex & x,const cuDoubleComplex & y) {
	return hipCsub(x,y);
}

__device__ inline cuDoubleComplex operator* (const double & a,const cuDoubleComplex & x) {
	return make_hipDoubleComplex (a*hipCreal(x), a*hipCimag(x));
}

__device__ inline cuDoubleComplex operator* (const cuDoubleComplex & x,const double & a) {
	return make_hipDoubleComplex (a*hipCreal(x), a*hipCimag(x));
}

__device__ inline cuDoubleComplex operator+ (const double & a,const cuDoubleComplex & x) {
	return make_hipDoubleComplex (a+hipCreal(x), hipCimag(x));
}

__device__ inline cuDoubleComplex operator+ (const cuDoubleComplex & x,const double & a) {
	return make_hipDoubleComplex (a+hipCreal(x), hipCimag(x));
}
*/
__device__ inline double Norm_2(const cuDoubleComplex & x) {
	return (hipCreal(x)*hipCreal(x)) + (hipCimag(x)*hipCimag(x));
}

__global__ void gpu_X_Equal_AX_minus_X_eigVal(double* Xtemp, double *AX, double *X, double *eigen, int len ,int bandLen)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int bid = tid / bandLen;
	if(tid < len)
	{
		Xtemp[tid] = AX[tid] - X[tid] * eigen[bid];
	}
}
__global__ void gpu_batch_Scal( double *psi, double *vec, int bandLen, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int iband = tid / bandLen;
	if(tid < len)
	{
		double alpha = 1.0 / sqrt( vec[iband] );
		psi[tid] = psi[tid] * alpha;
	}
}
	template<unsigned int blockSize>
__global__ void gpu_reduce( double * density, double * sum_den, int len)
{
	__shared__ double sdata[DIM];
	int offset = blockIdx.x * len;
	int tid = threadIdx.x;
	double s = 0.0;

	while ( tid < len)
	{
		int index = tid + offset;
		s += density[index];
		tid += blockDim.x;
	}

	sdata[threadIdx.x] =  s;
	double mySum = s;
	__syncthreads();

	tid = threadIdx.x;
	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 256];
	}

	__syncthreads();

	if ((blockSize >= 256) &&(tid < 128))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 128];
	}

	__syncthreads();

	if ((blockSize >= 128) && (tid <  64))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  64];
	}

	__syncthreads();
#if(0)
//#if (__CUDA_ARCH__ >= 300 )
	if ( tid < 32 )
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >=  64) mySum += sdata[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = warpSize/2; offset > 0; offset /= 2)
		{
			mySum += __shfl_down(mySum, offset);
		}
	}
#else
	// fully unroll reduction within a single warp
	if ((blockSize >=  64) && (tid < 32))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 32];
	}

	__syncthreads();

	if ((blockSize >=  32) && (tid < 16))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 16];
	}

	__syncthreads();

	if ((blockSize >=  16) && (tid <  8))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  8];
	}

	__syncthreads();

	if ((blockSize >=   8) && (tid <  4))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  4];
	}

	__syncthreads();

	if ((blockSize >=   4) && (tid <  2))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  2];
	}

	__syncthreads();

	if ((blockSize >=   2) && ( tid <  1))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  1];
	}

	__syncthreads();
#endif

	if( tid == 0) sum_den[blockIdx.x] = mySum;

}
	template<unsigned int blockSize>
__global__ void gpu_energy( double * psi, double * energy, int len)
{
	__shared__ double sdata[DIM];
	int offset = blockIdx.x * len;
	int tid = threadIdx.x;
	double s = 0.0;

	while ( tid < len)
	{
		int index = tid + offset;
		s += psi[index] * psi[index];
		tid += blockDim.x;
	}

	sdata[threadIdx.x] =  s;
	double mySum = s;
	__syncthreads();

	tid = threadIdx.x;
	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 256];
	}

	__syncthreads();

	if ((blockSize >= 256) &&(tid < 128))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 128];
	}

	__syncthreads();

	if ((blockSize >= 128) && (tid <  64))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  64];
	}

	__syncthreads();

#if (__CUDA_ARCH__ >= 300 )
	if ( tid < 32 )
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >=  64) mySum += sdata[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = warpSize/2; offset > 0; offset /= 2)
		{
			mySum += __shfl_down_sync(0xFFFFFFFF, mySum, offset);
		}
	}
#else
	// fully unroll reduction within a single warp
	if ((blockSize >=  64) && (tid < 32))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 32];
	}

	__syncthreads();

	if ((blockSize >=  32) && (tid < 16))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 16];
	}

	__syncthreads();

	if ((blockSize >=  16) && (tid <  8))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  8];
	}

	__syncthreads();

	if ((blockSize >=   8) && (tid <  4))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  4];
	}

	__syncthreads();

	if ((blockSize >=   4) && (tid <  2))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  2];
	}

	__syncthreads();

	if ((blockSize >=   2) && ( tid <  1))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  1];
	}

	__syncthreads();
#endif

	if( tid == 0) energy[blockIdx.x] = mySum;

}
__global__ void gpu_mapping_to_buf( double *buf, double * psi, int *index, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int x;
	if(tid < len)
	{
		x = index[tid];
		buf[x] = psi[tid];
	}
}

__global__ void gpu_mapping_from_buf( double *psi, double * buf, int *index, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int x;
	if(tid < len)
	{
		x = index[tid];
		psi[tid] = buf[x];
	}
}
//#if __CUDA_ARCH__ < 600
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600 

#else 
__device__ double atomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(val +
					__longlong_as_double(assumed)));

		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}
#endif
__global__ void gpu_setValue( float* dev, float val, int len)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < len)
		dev[tid] = val;
}

__global__ void gpu_setValue( double* dev, double val, int len)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < len)
		dev[tid] = val;
}
__global__ void gpu_setValue( cuDoubleComplex* dev, cuDoubleComplex val, int len)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < len){
		dev[tid].x = val.x;
		dev[tid].y = val.y;
	}
}
__global__ void gpu_setValue( cuComplex* dev, cuComplex val, int len)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < len){
		dev[tid].x = val.x;
		dev[tid].y = val.y;
	}
}
__global__ void  gpu_interpolate_wf_C2F( cuDoubleComplex * coarse_psi, cuDoubleComplex* fine_psi, int *index, int len, double factor)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < len)
	{
		int idx = index[tid];
		fine_psi[idx] = coarse_psi[tid] * factor;
	}
}

__global__ void  gpu_interpolate_wf_F2C( cuDoubleComplex * fine_psi, cuDoubleComplex* coarse_psi, int *index, int len, double factor)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < len)
	{
		int idx = index[tid];
		coarse_psi[tid] = coarse_psi[tid] + fine_psi[idx] * factor;
	}
}
__global__ void gpu_laplacian ( cuDoubleComplex * psi, double * gkk, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < len)
	{
		psi[tid] = psi[tid] * gkk[tid];
	}
}
__global__ void gpu_teter( cuDoubleComplex * psi, double * teter, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < len)
	{
		psi[tid] = psi[tid] * teter[tid];
	}
}

__global__ void gpu_vtot( double* psi, double * gkk, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < len)
	{
		psi[tid] = psi[tid] * gkk[tid];
	}
}
__global__ void gpu_update_psiUpdate( double *psiUpdate, double* NL, int * parts, int *index, double *atom_weight, double* weight)
{
	int start = parts[blockIdx.x];
	int end   = parts[blockIdx.x+1];
	int len   = end - start;
	int tid = threadIdx.x;
	double w = weight[blockIdx.x] * atom_weight[blockIdx.x];
	double s;

	while(tid < len)
	{
		int j = start + tid;
		int i = index[j];
		s = NL[j] * w;
		atomicAdd(&psiUpdate[i], s);
		tid += blockDim.x;
	}
}

	template<unsigned int blockSize>
__global__ void gpu_cal_weight( double * psi, double * NL, int * parts, int * index, double * weight)
{
	//first get the starting and ending point and length
	__shared__ double sdata[DIM];
	int start = parts[blockIdx.x];
	int end   = parts[blockIdx.x+1];
	int len   = end - start;

	int tid   = threadIdx.x;
	double s = 0.0;
	while(tid < len)
	{
		int j = start + tid;
		int i = index[j];
		s += psi[i] * NL[j];
		tid += blockDim.x;
	}

	sdata[threadIdx.x] =  s;
	double mySum = s;
	__syncthreads();

	tid = threadIdx.x;
	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 256];
	}

	__syncthreads();

	if ((blockSize >= 256) &&(tid < 128))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 128];
	}

	__syncthreads();

	if ((blockSize >= 128) && (tid <  64))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  64];
	}

	__syncthreads();

#if (__CUDA_ARCH__ >= 300 )
	if ( tid < 32 )
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >=  64) mySum += sdata[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = warpSize/2; offset > 0; offset /= 2)
		{
			mySum += __shfl_down_sync(0xFFFFFFFF, mySum, offset);
		}
	}
#else
	// fully unroll reduction within a single warp
	if ((blockSize >=  64) && (tid < 32))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 32];
	}

	__syncthreads();

	if ((blockSize >=  32) && (tid < 16))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 16];
	}

	__syncthreads();

	if ((blockSize >=  16) && (tid <  8))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  8];
	}

	__syncthreads();

	if ((blockSize >=   8) && (tid <  4))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  4];
	}

	__syncthreads();

	if ((blockSize >=   4) && (tid <  2))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  2];
	}

	__syncthreads();

	if ((blockSize >=   2) && ( tid <  1))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  1];
	}

	__syncthreads();
#endif 

	if( tid == 0) weight[blockIdx.x] = mySum;
}
#if 0
float* cuda_malloc( float* ptr, size_t size)
{
	printf("hipMalloc float the pointer %d \n", size);
	CUDA_CALL( hipMalloc( &ptr, size ) );
	return ptr;
}
double* cuda_malloc( double* ptr, size_t size)
{
	printf("hipMalloc double the pointer %d \n", size);
	CUDA_CALL( hipMalloc( &ptr, size ) );
	return ptr;
}
cuComplex* cuda_malloc( cuComplex* ptr, size_t size)
{
	printf("hipMalloc complex the pointer %d \n", size);
	CUDA_CALL( hipMalloc( &ptr, size ) );
	return ptr;
}
cuDoubleComplex* cuda_malloc( cuDoubleComplex* ptr, size_t size)
{
	printf("hipMalloc double complex the pointer %d \n", size);
	CUDA_CALL( hipMalloc( &ptr, size ) );
	return ptr;
}
#endif


void cuda_free( void *ptr)
{
	//CUDA_CALL( hipFree(ptr) );
        hipFree(ptr);
        //std::cout << "lijl test "<< std::endl << std::endl;
}

void cuda_memcpy_CPU2GPU( void *gpu, void * cpu, size_t size )
{
	//CUDA_CALL( hipMemcpy(gpu, cpu, size, hipMemcpyHostToDevice ); );
        hipMemcpy(gpu, cpu, size, hipMemcpyHostToDevice );
	//std::flush(std::cout);
}

void cuda_memcpy_GPU2CPU( void *cpu, void * gpu, size_t size )
{
	//CUDA_CALL( hipMemcpy(cpu, gpu, size, hipMemcpyDeviceToHost); );
        hipMemcpy(cpu, gpu, size, hipMemcpyDeviceToHost); 
}

void cuda_memcpy_GPU2GPU( void * dest, void * src, size_t size)
{
	//CUDA_CALL( hipMemcpy(dest, src, size, hipMemcpyDeviceToDevice); );
        hipMemcpy(dest, src, size, hipMemcpyDeviceToDevice);
}

void cuda_setValue( float* dev, float val, int len )
{
	int ndim = len / DIM;
	if(len % DIM) ndim++;
	hipLaunchKernelGGL((gpu_setValue), dim3(ndim), dim3(DIM), 0, 0, dev, val, len);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}

void cuda_setValue( double* dev, double val, int len )
{
	int ndim = len / DIM;
	if(len % DIM) ndim++;
	hipLaunchKernelGGL((gpu_setValue), dim3(ndim), dim3(DIM), 0, 0, dev, val, len);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}
void cuda_setValue( cuDoubleComplex* dev, cuDoubleComplex val, int len )
{
	int ndim = len / DIM;
	if(len % DIM) ndim++;
	hipLaunchKernelGGL((gpu_setValue), dim3(ndim), dim3(DIM), 0, 0, dev, val, len);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}

void cuda_setValue( cuComplex* dev, cuComplex val, int len )
{
	int ndim = len / DIM;
	if(len % DIM) ndim++;
	hipLaunchKernelGGL((gpu_setValue), dim3(ndim), dim3(DIM), 0, 0, dev, val, len);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}

void cuda_interpolate_wf_C2F( cuDoubleComplex * coarse_psi, cuDoubleComplex * fine_psi, int * index, int len, double factor)
{
	int ndim = (len + DIM - 1) / DIM;
	hipLaunchKernelGGL((gpu_interpolate_wf_C2F), dim3(ndim), dim3(DIM), 0, 0,  coarse_psi, fine_psi, index, len, factor);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}
void cuda_interpolate_wf_F2C( cuDoubleComplex * fine_psi, cuDoubleComplex * coarse_psi, int * index, int len, double factor)
{
	int ndim = (len + DIM - 1) / DIM;
	hipLaunchKernelGGL((gpu_interpolate_wf_F2C), dim3(ndim), dim3(DIM), 0, 0,  fine_psi, coarse_psi, index, len, factor);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}

void *cuda_malloc( size_t size)
{
	void *ptr;
	//CUDA_CALL( hipMalloc( &ptr, size ) );
        hipMalloc( &ptr, size );
	return ptr;
}
void cuda_laplacian( cuDoubleComplex* psi, double * gkk, int len)
{
	int ndim = (len + DIM - 1) / DIM;
	hipLaunchKernelGGL((gpu_laplacian), dim3(ndim), dim3(DIM), 0, 0,  psi, gkk, len);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}
void cuda_vtot( double* psi, double * vtot, int len)
{
	int ndim = (len + DIM - 1) / DIM;
	hipLaunchKernelGGL((gpu_vtot), dim3(ndim), dim3(DIM), 0, 0,  psi, vtot, len);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}

void cuda_memory(void)
{
	size_t free_mem, total_mem;
	hipMemGetInfo(&free_mem, &total_mem);
	hipMemGetInfo(&free_mem, &total_mem);
	assert(hipDeviceSynchronize() == hipSuccess );
	printf("free  memory is: %zu MB\n", free_mem/1000000);
	printf("total memory is: %zu MB\n", total_mem/1000000);
	fflush(stdout);
} 

void cuda_calculate_nonlocal( double * psiUpdate, double * psi, double * NL, int * index, int * parts,  double * atom_weight, double * weight, int blocks)
{
	// two steps. 
	// 1. calculate the weight.
	hipLaunchKernelGGL((gpu_cal_weight<DIM>), dim3(blocks), dim3(DIM), DIM * sizeof(double) , 0,  psi, NL, parts, index, weight);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif

	// 2. update the psiUpdate.
	hipLaunchKernelGGL((gpu_update_psiUpdate), dim3(blocks), dim3(LDIM), 0, 0,  psiUpdate, NL, parts, index, atom_weight, weight);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}
void cuda_teter( cuDoubleComplex* psi, double * vtot, int len)
{
	int ndim = (len + DIM - 1) / DIM;
	hipLaunchKernelGGL((gpu_teter), dim3(ndim), dim3(DIM), 0, 0,  psi, vtot, len);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}

void cuda_mapping_from_buf( double * psi, double * buf, int * index, int len )
{
	int ndim = (len + DIM - 1) / DIM;
	hipLaunchKernelGGL((gpu_mapping_from_buf), dim3(ndim), dim3(DIM), 0, 0,  psi, buf, index, len);	
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}

void cuda_mapping_to_buf( double * buf, double * psi, int * index, int len )
{
	int ndim = (len + DIM - 1) / DIM;
	hipLaunchKernelGGL((gpu_mapping_to_buf), dim3(ndim), dim3(DIM), 0, 0,  buf, psi, index, len);	
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}

void cuda_reduce( double * density, double * sum, int nbands, int bandLen)
  {
         hipLaunchKernelGGL((gpu_reduce<DIM>), dim3(nbands), dim3(DIM), DIM*sizeof(double), 0,  density, sum, bandLen);
  #ifdef SYNC 
          gpuErrchk(hipPeekAtLastError());
          gpuErrchk(hipDeviceSynchronize());
          assert(hipThreadSynchronize() == hipSuccess );
  #endif
  }
void cuda_calculate_Energy( double * psi, double * energy, int nbands, int bandLen)
{
	// calculate  nbands psi Energy. 
	hipLaunchKernelGGL((gpu_energy<DIM>), dim3(nbands), dim3(DIM), DIM*sizeof(double), 0,  psi, energy, bandLen);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}

void cuda_batch_Scal( double * psi, double * vec, int nband, int bandLen)
{
	int ndim = ( nband * bandLen + DIM - 1) / DIM;
	int len = nband * bandLen;
	hipLaunchKernelGGL((gpu_batch_Scal), dim3(ndim), dim3(DIM ), 0, 0,  psi, vec, bandLen, len);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}
void cu_X_Equal_AX_minus_X_eigVal( double * Xtemp, double * AX, double * X, double * eigen, int nbands, int bandLen)
{
	int ndim = ( nbands * bandLen + DIM - 1 ) / DIM;
	int len = nbands * bandLen;
	hipLaunchKernelGGL((gpu_X_Equal_AX_minus_X_eigVal), dim3(ndim), dim3(DIM ), 0, 0,  Xtemp, AX, X, eigen, len, bandLen );
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}

void cuda_init_vtot()
{
	dev_vtot           = NULL;
	dev_gkkR2C         = NULL;
	dev_idxFineGridR2C = NULL;
	dev_NLindex        = NULL;
	dev_NLpart         = NULL;
	dev_NLvecFine      = NULL;
	dev_atom_weight    = NULL;
	dev_temp_weight    = NULL;
	dev_TeterPrecond   = NULL;
	vtot_gpu_flag      = false;
	NL_gpu_flag        = false;
	teter_gpu_flag     = false;
}
void cuda_clean_vtot()
{
	cuda_free(dev_vtot);
	cuda_free(dev_gkkR2C);
	cuda_free(dev_idxFineGridR2C);
	cuda_free(dev_NLindex);
	cuda_free(dev_NLpart);
	cuda_free(dev_NLvecFine);
	cuda_free(dev_atom_weight);
	cuda_free(dev_temp_weight);
	cuda_free(dev_TeterPrecond);
}
void cuda_set_vtot_flag()
{
	vtot_gpu_flag  = false;
}

__global__ void gpu_matrix_add( double * A, double * B, int length )
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid < length )
	{
		A[tid] = A[tid] + B[tid];
	}
}

void cuda_DMatrix_Add( double * A , double * B, int m, int n)
{
	// Matrix A and B are double dimension m,n; A = A + B
	int ndim = (m * n + DIM - 1) / DIM;
	int length = m*n;
	hipLaunchKernelGGL((gpu_matrix_add), dim3(ndim), dim3(DIM ), 0, 0,  A, B, length);

#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}
__global__ void gpu_alpha_X_plus_beta_Y_multiply_Z( double * X, double alpha, double * Y, double beta, double * Z, int length)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < length)
	{
		X[tid] = alpha * X[tid] + beta * Y[tid] * Z[tid];
	}
}
void cuda_Axpyz( double * X, double alpha, double * Y, double beta, double * Z, int length)
{
	int ndim = ( length + DIM - 1) / DIM ;
	hipLaunchKernelGGL((gpu_alpha_X_plus_beta_Y_multiply_Z), dim3(ndim), dim3(DIM ), 0, 0, X, alpha, Y, beta, Z, length);

#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}

__global__ void gpu_cal_sendk( int *sendk, int * senddisps, int widthLocal, int height, int heightBlockSize, int mpisize)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid < height* widthLocal)
	{
		int i = tid % height;
		int j = tid / height;

		if(height % mpisize == 0){
			sendk[tid] = senddisps[i/heightBlockSize] + j * heightBlockSize + i % heightBlockSize;
		}
		else{
			if( i < ((height%mpisize) * (heightBlockSize + 1)) ) {
				sendk[tid] = senddisps[i/(heightBlockSize + 1)] + j * ( heightBlockSize + 1) + i % ( heightBlockSize + 1);
			}
			else{
				sendk[tid] = senddisps[(height % mpisize) + (i-(height % mpisize)*(heightBlockSize+1))/heightBlockSize] 
					+ j * heightBlockSize + (i-(height % mpisize)*(heightBlockSize+1)) % heightBlockSize;
			}
		}
	}
}

__global__ void gpu_cal_recvk( int *recvk, int * recvdisps, int width, int heightLocal, int mpisize)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid < heightLocal* width)
	{
		int i = tid % heightLocal;
		int j = tid / heightLocal;

		recvk[tid] = recvdisps[j%mpisize] + ( j/mpisize) * heightLocal + i;
	}
}

void cuda_cal_sendk( int * sendk, int * senddispl, int widthLocal, int height, int heightBlockSize, int mpisize)
{
	int total = widthLocal * height;
	int dim = (total + LEN - 1) / LEN;
	
       	hipLaunchKernelGGL((gpu_cal_sendk), dim3(dim), dim3(LEN), 0, 0,  sendk, senddispl, widthLocal, height, heightBlockSize, mpisize );
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}

void cuda_cal_recvk( int * recvk, int * recvdisp, int width, int heightLocal, int mpisize)
{
	int total = width * heightLocal;
	int dim = ( total + LEN - 1 ) / LEN;
	
	hipLaunchKernelGGL((gpu_cal_recvk), dim3(dim), dim3(LEN), 0, 0,  recvk, recvdisp, width, heightLocal, mpisize );
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}
template < class T >
__global__ void gpu_hadamard_product ( T* dev_A, T* dev_B, T * dev_result, int length)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < length)
	{
		dev_result[tid] = dev_A[tid] * dev_B[tid];
	}
}

void cuda_hadamard_product( double * in1, double * in2, double * out, int length)
{
	int dim = ( length + LEN - 1 ) / LEN;
	hipLaunchKernelGGL((gpu_hadamard_product<double>), dim3(dim), dim3(LEN ), 0, 0,  in1, in2, out, length );

#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}
template <class T>
__global__ void gpu_set_vector( T* out, T* in , int length)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid < length)
	{
		out[tid] = in[tid];
	}
}

void cuda_set_vector( double * out, double *in, int length)
{
	int dim = (length + LEN - 1) / LEN;
	hipLaunchKernelGGL((gpu_set_vector< double>), dim3(dim), dim3(LEN ), 0, 0,  out, in, length);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}
__global__ void gpu_XTX ( cuDoubleComplex *X, double * Y, int len)
  {                                                               
          int tid = blockIdx.x * blockDim.x + threadIdx.x;        
 	  cuDoubleComplex x;
	  double y; 
          if(tid < len)         
	  {                                     
                  x = X[tid];                   
                  y = x.x;                        
		  //y.y = - x.y;
                  Y[tid] = y*y;                     
	  }                    
  } 

void cuda_XTX( cuDoubleComplex * X, double * Y, int length)                
{                                                                   
  	int dim = ( length + LEN - 1) / LEN;                                  
          hipLaunchKernelGGL((gpu_XTX), dim3(dim), dim3(LEN), 0, 0, X, Y, length);                              
  #ifdef SYNC                                                                 
          gpuErrchk(hipPeekAtLastError());                                   
          gpuErrchk(hipDeviceSynchronize());                                 
          assert(hipThreadSynchronize() == hipSuccess );                    
  #endif
}
//-------------------by lijl 20210602
__global__ void gpu_X_Equal_XP(double * X, double * P, int len, int row)
{
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if( tid < len )
        {
            int index = tid % row;
            X[tid] *= P[index];            
        }
}

void cuda_X_Equal_XP( double * X, double * P, int row, int col)
{
        int ndim = (row * col + DIM - 1) / DIM;
        int len = row * col;
        hipLaunchKernelGGL((gpu_X_Equal_XP), dim3(ndim), dim3(DIM), 0, 0, X, P, len, row);
#ifdef SYNC 
          gpuErrchk(hipPeekAtLastError());                                   
          gpuErrchk(hipDeviceSynchronize());                                 
          assert(hipThreadSynchronize() == hipSuccess );                    
#endif
}

__global__ void gpu_FourDotProduct( double * BasisTemp, double * LGLWeight, double * vtot, double * Basis, int len, int row)
{
         int tid = blockIdx.x * blockDim.x + threadIdx.x;
         if( tid < len )
         {
             int index = tid % row;
             BasisTemp[tid] = LGLWeight[index] * vtot[index] * Basis[tid];
         }
}

void cuda_FourDotProduct(double * BasisTemp, double * LGLWeight, double * vtot, double * Basis, int row, int col)
{
        int ndim = (row * col + DIM - 1) / DIM;
        int len = row * col;
        hipLaunchKernelGGL((gpu_FourDotProduct), dim3(ndim), dim3(DIM), 0, 0 , BasisTemp, LGLWeight, vtot, Basis, len, row);
#ifdef SYNC 
          gpuErrchk(hipPeekAtLastError());                                   
          gpuErrchk(hipDeviceSynchronize());                                 
          assert(hipThreadSynchronize() == hipSuccess );                    
#endif
}

__global__ void gpu_XTemp_Equal_XP(double * XTemp, double * X, double * P, int len, int row)
{
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if( tid < len )
        {
            int index = tid % row;
            XTemp[tid] = X[tid] * P[index];            
        }
}

void cuda_XTemp_Equal_XP( double * XTemp, double *X, double * P, int row, int col)
{
        int ndim = (row * col + DIM - 1) / DIM;
        int len = row * col;
        hipLaunchKernelGGL((gpu_XTemp_Equal_XP), dim3(ndim), dim3(DIM), 0, 0, XTemp, X, P, len, row);
#ifdef SYNC 
          gpuErrchk(hipPeekAtLastError());                                   
          gpuErrchk(hipDeviceSynchronize());                                 
          assert(hipThreadSynchronize() == hipSuccess );                    
#endif
}

__global__ void gpu_localMat2(double * localMatTemp4, double * localMatTemp3, double * localMatTemp2, double *localMatTemp1, double penaltyAlpha, int len)
{
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if( tid < len )
        {
            localMatTemp4[tid] = 0.0 - 0.5 * localMatTemp1[tid] - 0.5 * localMatTemp2[tid] + penaltyAlpha * localMatTemp3[tid];
        }
}

void cuda_localMat2(double * localMatTemp4, double * localMatTemp3, double * localMatTemp2, double *localMatTemp1, double penaltyAlpha, int row, int col)
{
        int ndim = (row * col + DIM - 1) / DIM;
        int len = row * col;
        hipLaunchKernelGGL((gpu_localMat2), dim3(ndim), dim3(DIM), 0, 0, localMatTemp4, localMatTemp3, localMatTemp2, localMatTemp1, penaltyAlpha, len);
#ifdef SYNC 
          gpuErrchk(hipPeekAtLastError());                                   
          gpuErrchk(hipDeviceSynchronize());                                 
          assert(hipThreadSynchronize() == hipSuccess );                    
#endif
}

__global__ void gpu_localMat(double * localMatTemp7, double * localMatTemp6, double * localMatTemp5, double * localMatTemp4, double * localMatTemp3, double * localMatTemp2, double *localMatTemp1, double penaltyAlpha, int len)
{
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if( tid < len )
        {
            localMatTemp7[tid] = 0.0 - 0.5 * localMatTemp1[tid] - 0.5 * localMatTemp2[tid] - 0.5 * localMatTemp3[tid] - 0.5 * localMatTemp4[tid] + penaltyAlpha * localMatTemp5[tid] + penaltyAlpha * localMatTemp6[tid];
        }
}

void cuda_localMat(double * localMatTemp7, double * localMatTemp6, double * localMatTemp5, double * localMatTemp4, double * localMatTemp3, double * localMatTemp2, double *localMatTemp1, double penaltyAlpha, int row, int col)
{
        int ndim = (row * col + DIM - 1) / DIM;
        int len = row * col;
        hipLaunchKernelGGL((gpu_localMat), dim3(ndim), dim3(DIM), 0, 0, localMatTemp7, localMatTemp6, localMatTemp5, localMatTemp4, localMatTemp3, localMatTemp2, localMatTemp1, penaltyAlpha, len);
#ifdef SYNC 
          gpuErrchk(hipPeekAtLastError());                                   
          gpuErrchk(hipDeviceSynchronize());                                 
          assert(hipThreadSynchronize() == hipSuccess );                    
#endif
}

__global__ void gpu_InnerProduct(double * basis, double * coef, double * weight, int * Idx, double * Val, int * basisLGLIdx, int numLGLGrid,  int idxSize)
{
        int tid = threadIdx.x; // tid denotes idxSize
        int bid = blockIdx.x;  // bid denotes numBasis
        __shared__ double shared_product[BLOCK_SIZE];
        shared_product[threadIdx.x] = 0;
        int a = basisLGLIdx[bid];
        while( tid < idxSize )
        {
            int l = Idx[tid]; 
            double temp = weight[l] * basis[l + bid * numLGLGrid] * Val[tid];
            shared_product[threadIdx.x] += temp;
            tid += BLOCK_SIZE;
        }
        __syncthreads();
        
        for(size_t i = BLOCK_SIZE >> 1; i >= 1; i = i >> 1)
        {
            if(threadIdx.x < i)
            {
                shared_product[threadIdx.x] += shared_product[threadIdx.x + i];
            }
            __syncthreads();
        }
        if(threadIdx.x == 0)
        {
            coef[a] += shared_product[threadIdx.x];
        }
}

void cuda_InnerProduct(double * basis, double * coef, double * weight, int * Idx, double * Val, int * basisLGLIdx, int numLGLGrid, int numBasis, int idxSize)
{
        hipLaunchKernelGGL((gpu_InnerProduct), dim3(numBasis), dim3(BLOCK_SIZE), 0, 0, basis, coef, weight, Idx, Val, basisLGLIdx, numLGLGrid, idxSize);
#ifdef SYNC 
          gpuErrchk(hipPeekAtLastError());                                   
          gpuErrchk(hipDeviceSynchronize());                                 
          assert(hipThreadSynchronize() == hipSuccess );                    
#endif
}

__global__ void gpu_sqrt(double * S, int numBasis)
{
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < numBasis)
        {
            S[tid] = sqrt( S[tid] );
        }
}

void cuda_sqrt(double * S, int numBasis)
{
        int ndim = (numBasis + DIM - 1) / DIM;
        hipLaunchKernelGGL((gpu_sqrt), dim3(ndim), dim3(DIM), 0, 0, S, numBasis);
#ifdef SYNC 
          gpuErrchk(hipPeekAtLastError());                                   
          gpuErrchk(hipDeviceSynchronize());                                 
          assert(hipThreadSynchronize() == hipSuccess );                    
#endif
}

// --- xmqin 20220714
__global__ void gpu_sqrt(double *odata, double *idata, int len)
{
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < len)
        {
            odata[tid] = sqrt( idata[tid] );
        }
}

void cuda_sqrt(double *odata, double *idata,  int len)
{
        int ndim = (len + DIM - 1) / DIM;
        hipLaunchKernelGGL((gpu_sqrt), dim3(ndim), dim3(DIM), 0, 0, odata, idata, len);
#ifdef SYNC
          gpuErrchk(hipPeekAtLastError());
          gpuErrchk(hipDeviceSynchronize());
          assert(hipThreadSynchronize() == hipSuccess );
#endif
}
//------

__global__ void gpu_division(double * Basis, double * LGLWeight, int len, int row)
{
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if( tid < len )
        {
            int index = tid % row;
            Basis[tid] = Basis[tid] / LGLWeight[index];
        }
}

void cuda_division(double * Basis, double * LGLWeight, int row, int col)
{
        int ndim = (row * col + DIM - 1) / DIM;
        int len = row * col;
        hipLaunchKernelGGL((gpu_division), dim3(ndim), dim3(DIM), 0, 0, Basis, LGLWeight, len, row);
#ifdef SYNC 
          gpuErrchk(hipPeekAtLastError());                                   
          gpuErrchk(hipDeviceSynchronize());                                 
          assert(hipThreadSynchronize() == hipSuccess );                    
#endif
}
#if 1
__global__ void gpu_RhoLGL(double * localRhoLGLTmp, double * localBasisRow, double * localDM, int numBasisTotal, int heightLocal, int idxSta)
{
        int tid = threadIdx.x;  // tid denotes a * b (b > = a), size : (numBasisTotal + 1) * numBasisTotal
        int bid = blockIdx.x; // bid denotes p, size : heightLocal
        __shared__ double shared_product[BLOCK_SIZE];
        shared_product[threadIdx.x] = 0;
        int len = (numBasisTotal + 1) * numBasisTotal / 2;
        while( tid < len )
        {
          int a = -1;
          int Res;
          int N = numBasisTotal;
          int tid_temp = tid;
          while(tid_temp >= 0)
          {
              Res = tid_temp % N;
              tid_temp = tid_temp - N;
              N--;
              a++;
          }
          int b = a + Res;
          //int a = cu_IndexA[tid];
          //int b = cu_IndexB[tid];
          double factor = localDM[a + numBasisTotal * b];
          factor += factor*int(b>a);
          double temp = localBasisRow[bid + heightLocal * a] * localBasisRow[bid + heightLocal * b] * factor;
          shared_product[threadIdx.x] += temp;
          tid += BLOCK_SIZE;
        } 
        __syncthreads();

       for(size_t i = BLOCK_SIZE >> 1; i >= 1; i = i >> 1)
       {
           if(threadIdx.x < i)
           {
               shared_product[threadIdx.x] += shared_product[threadIdx.x + i];
           }
           __syncthreads();
       }
      
      if(threadIdx.x == 0)
      {
          localRhoLGLTmp[bid + idxSta] = shared_product[threadIdx.x];
      }
}

void cuda_RhoLGL(double * localRhoLGLTmp, double * localBasisRow, double * localDM,  int numBasisTotal, int heightLocal, int idxSta)
{
        
        hipLaunchKernelGGL((gpu_RhoLGL), dim3(heightLocal), dim3(BLOCK_SIZE), 0, 0, localRhoLGLTmp, localBasisRow, localDM, numBasisTotal, heightLocal, idxSta);
#ifdef SYNC 
          gpuErrchk(hipPeekAtLastError());                                   
          gpuErrchk(hipDeviceSynchronize());                                 
          assert(hipThreadSynchronize() == hipSuccess );                    
#endif

}
#endif
#if 0
__global__ void gpu_Gemv_batched( double *A, int m, int n, int k, int inc, double * tmp1, double * tmp2)
void cuda_Gemv_batched (double * A, int m, int n, int k, int inc, double * tmp1, double * tmp2 )
{
         int dim = ( length + LEN - 1) / LEN;
         gpu_Gemv_batched<<< dim, LEN>>>( A, m, n, k, inc, tmp1, tmp2)
   #ifdef SYNC                                                                 
          gpuErrchk(cudaPeekAtLastError());
          gpuErrchk(cudaDeviceSynchronize());
          assert(cudaThreadSynchronize() == hipSuccess );
  #endif
}
//----------------------by lijl 20210602
#endif

void cuda_sync()
{
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
}
#endif

void cuda_memcpyAsync_CPU2GPU( const hipStream_t stream, void *gpu, void *cpu, size_t size )
{
       hipMemcpyAsync(gpu, cpu, size, hipMemcpyHostToDevice, stream );
}

void cuda_memcpyAsync_GPU2CPU( const hipStream_t stream, void *cpu, void * gpu, size_t size )
{
       hipMemcpyAsync(cpu, gpu, size, hipMemcpyDeviceToHost, stream);
}

void cuda_memcpyAsync_GPU2GPU( const hipStream_t stream, void * dest, void * src, size_t size)
{
       hipMemcpyAsync(dest, src, size, hipMemcpyDeviceToDevice, stream); 
}


void cuda_setValue_stream(const hipStream_t stream, float* dev, float val, int len )
{
        int ndim = len / DIM;
        if(len % DIM) ndim++;
//        gpu_setValue<<<ndim, DIM, 0, stream>>>(dev, val, len);
        hipLaunchKernelGGL((gpu_setValue), dim3(ndim), dim3(DIM), 0, stream, dev, val, len);
#ifdef SYNC
        gpuErrchk(hipPeekAtLastError());
        gpuErrchk(hipDeviceSynchronize());
        assert(hipDeviceSynchronize() == cudaSuccess );
#endif
}

void cuda_setValue_stream( const hipStream_t stream, double* dev, double val, int len )
{
        int ndim = len / DIM;
        if(len % DIM) ndim++;
  //      gpu_setValue<<<ndim, DIM, 0, stream>>>(dev, val, len);
        hipLaunchKernelGGL((gpu_setValue), dim3(ndim), dim3(DIM), 0, stream, dev, val, len);
#ifdef SYNC
        gpuErrchk(hipPeekAtLastError());
        gpuErrchk(hipDeviceSynchronize());
        assert(hipDeviceSynchronize() == cudaSuccess );
#endif
}
void cuda_setValue_stream( const hipStream_t stream, cuDoubleComplex* dev, cuDoubleComplex val, int len )
{
        int ndim = len / DIM;
        if(len % DIM) ndim++;
//        gpu_setValue<<<ndim, DIM, 0, stream>>>(dev, val, len);
        hipLaunchKernelGGL((gpu_setValue), dim3(ndim), dim3(DIM), 0, stream, dev, val, len);
#ifdef SYNC
        gpuErrchk(hipPeekAtLastError());
        gpuErrchk(hipDeviceSynchronize());
        assert(hipDeviceSynchronize() == cudaSuccess );
#endif
}
void cuda_setValue_stream ( const hipStream_t stream, cuComplex* dev, cuComplex val, int len )
{
        int ndim = len / DIM;
        if(len % DIM) ndim++;
//        gpu_setValue<<<ndim, DIM, 0, stream>>>(dev, val, len);
        hipLaunchKernelGGL((gpu_setValue), dim3(ndim), dim3(DIM), 0, stream, dev, val, len);
#ifdef SYNC
        gpuErrchk(hipPeekAtLastError());
        gpuErrchk(hipDeviceSynchronize());
        assert(hipDeviceSynchronize() == cudaSuccess );
#endif
}

void cuda_interpolate_wf_C2F_stream( const hipStream_t stream, cuDoubleComplex * coarse_psi, cuDoubleComplex * fine_psi, int * index, int len, double factor)
{
        int ndim = (len + DIM - 1) / DIM;
        hipLaunchKernelGGL((gpu_interpolate_wf_C2F), dim3(ndim), dim3(DIM), 0, 0,  coarse_psi, fine_psi, index, len, factor);
#ifdef SYNC
        gpuErrchk(hipPeekAtLastError());
        gpuErrchk(hipDeviceSynchronize());
        assert(hipDeviceSynchronize() == cudaSuccess );
#endif
}


void cuda_interpolate_wf_F2C_stream( const hipStream_t stream, cuDoubleComplex * fine_psi, cuDoubleComplex * coarse_psi, int * index, int len, double factor)
{
        int ndim = (len + DIM - 1) / DIM;
        hipLaunchKernelGGL((gpu_interpolate_wf_F2C), dim3(ndim), dim3(DIM), 0, 0,  fine_psi, coarse_psi, index, len, factor);
#ifdef SYNC
        gpuErrchk(hipPeekAtLastError());
        gpuErrchk(hipDeviceSynchronize());
        assert(hipDeviceSynchronize() == cudaSuccess );
#endif
}


void cuda_laplacian_stream( const hipStream_t stream, cuDoubleComplex* psi, double * gkk, int len)
{
        int ndim = (len + DIM - 1) / DIM;
        hipLaunchKernelGGL((gpu_laplacian), dim3(ndim), dim3(DIM), 0, stream,  psi, gkk, len);
#ifdef SYNC
        gpuErrchk(hipPeekAtLastError());
        gpuErrchk(hipDeviceSynchronize());
        assert(hipDeviceSynchronize() == cudaSuccess );
#endif
}

void cuda_vtot_stream( const hipStream_t stream, double* psi, double * vtot, int len)
{
        int ndim = (len + DIM - 1) / DIM;
        hipLaunchKernelGGL((gpu_vtot), dim3(ndim), dim3(DIM), 0, stream,  psi, vtot, len);
#ifdef SYNC
        gpuErrchk(hipPeekAtLastError());
        gpuErrchk(hipDeviceSynchronize());
        assert(hipDeviceSynchronize() == cudaSuccess );
#endif
}

void cuda_calculate_nonlocal_stream( const hipStream_t stream, double * psiUpdate, double * psi, double * NL, int * index, int * parts,  double * atom_weight, double * weight, int blocks)
{
        // two steps.
        // 1. calculate the weight.
//        gpu_cal_weight<DIM><<<blocks, DIM, DIM * sizeof(double), stream >>>( psi, NL, parts, index, weight);
        hipLaunchKernelGGL((gpu_cal_weight<DIM>), dim3(blocks), dim3(DIM), DIM * sizeof(double) , stream,  psi, NL, parts, index, weight);

#ifdef SYNC
        gpuErrchk(hipPeekAtLastError());
        gpuErrchk(hipDeviceSynchronize());
        assert(hipDeviceSynchronize() == cudaSuccess );
#endif

        // 2. update the psiUpdate.
        hipLaunchKernelGGL((gpu_update_psiUpdate), dim3(blocks), dim3(LDIM), 0, stream,  psiUpdate, NL, parts, index, atom_weight, weight);
#ifdef SYNC
        gpuErrchk(hipPeekAtLastError());
        gpuErrchk(hipDeviceSynchronize());
        assert(hipDeviceSynchronize() == cudaSuccess );
#endif
}

void cuda_teter_stream( const hipStream_t stream, cuDoubleComplex* psi, double * vtot, int len)
{
        int ndim = (len + DIM - 1) / DIM;
        hipLaunchKernelGGL((gpu_teter), dim3(ndim), dim3(DIM), 0, stream,  psi, vtot, len);
#ifdef SYNC
        gpuErrchk(hipPeekAtLastError());
        gpuErrchk(hipDeviceSynchronize());
        assert(hipDeviceSynchronize() == cudaSuccess );
#endif
}

//----------- add by lijl 20220706 
//------edit by xmqin 20220714
#define BLOCK_DIM 16
__global__ void  transpose3D_revised_xyz2yxz( double *odata, double *idata, 
    unsigned int n1, unsigned int n2, unsigned int n3, 
    unsigned int Gx, unsigned int Gy, 
    float one_over_Gx, float one_over_Gy, unsigned int k2 ) ;
				
void  transpose3D_revised_xyz2yxz_device( double *Y, double *X, 
    unsigned int n1, unsigned int n2, unsigned int n3 ) 
			
{
    unsigned int Gx, Gy, k1, k2 ;
    double db_n3 = (double) n3 ;
	
    // we only accept out-of-place 
    if ( X == Y ){
        printf("Error(transpose3D_xyz2yxz_device): we only accept out-of-place \n");
        exit(1) ;
    }	

    Gx = (n1 + BLOCK_DIM-1) / BLOCK_DIM ; 
    Gy = (n2 + BLOCK_DIM-1) / BLOCK_DIM ; 
	
    int max_k2 = (int) floor( sqrt(db_n3) ) ;
    for ( k2 = max_k2 ; 1 <= k2 ; k2-- ){
        k1 = (unsigned int) ceil( db_n3/((double)k2)) ;
        if ( 1 >= (k1*k2 - n3) ){
            break ;
        }
    }

//   printf(" n1 = %d, n2 = %d, n3 = %d\n", n1, n2, n3);
//   printf(" BLOCK_DIM =  %d\n", BLOCK_DIM);
//   printf(" Gx = %d, Gy = %d\n", Gx, Gy);
//    printf(" k1 = %d, k2 = %d\n", k1, k2);
	
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 grid( k2*Gx, k1*Gy, 1 );
  
    float eps = 1.E-5 ;
    float one_over_Gx = (1.0 + eps)/((double)Gx) ; 
    float one_over_Gy = (1.0 + eps)/((double)Gy) ;
   
//    printf(" over_Gx = %d, over_Gy = %d\n",one_over_Gx,one_over_Gy);
//    printf(" Grid( %d,  %d )\n",k1*Gx, k2*Gy);

  
    hipLaunchKernelGGL((transpose3D_revised_xyz2yxz), grid, threads, BLOCK_DIM*(BLOCK_DIM+1)*sizeof(double), 0, Y, X, 
        n1, n2, n3, Gx, Gy, one_over_Gx, one_over_Gy, k2 ) ;
}

__global__ void  transpose3D_revised_xyz2yxz( double *odata, double *idata, 
    unsigned int n1, unsigned int n2, unsigned int n3, 
    unsigned int Gx, unsigned int Gy, 
    float one_over_Gx, float one_over_Gy, unsigned int k2 )
{
    __shared__ double block[BLOCK_DIM][BLOCK_DIM+1];

    float tmp1 ;
    unsigned int s1, s2, t1, t2 ;
    unsigned int xIndex, yIndex, zIndex ;
    unsigned int index_in, index_out ;
	
    tmp1 = __uint2float_rz( blockIdx.x ) ;
    tmp1 = floorf( tmp1 * one_over_Gx ) ;
    s1 = __float2uint_rz( tmp1 ) ; 
    t1 = blockIdx.x - Gx*s1 ;
 	
    tmp1 = __uint2float_rz( blockIdx.y ) ;
    tmp1 = floorf( tmp1 * one_over_Gy ) ;
    s2 = __float2uint_rz( tmp1 ) ; 
    t2 = blockIdx.y - Gy*s2 ;

 //   printf(" (s1=%d,  s2=%d )\n",s1, s2);
//    printf(" (t1=%d,  t2=%d )\n",t1, t2);
 
    zIndex = s2*k2 + s1 ;
 
    xIndex = t1 * BLOCK_DIM + threadIdx.x ;
    yIndex = t2 * BLOCK_DIM + threadIdx.y ;
//    printf(" (xIndex=%d,  yIndex=%d, zIndex = %d )\n",xIndex, yIndex, zIndex);

//        ind_in = x_in + (y_in + z * np1) * np0;
//        ind_out = y_out + (x_out + z * np0) * np1;

    if ( (zIndex < n3) && (yIndex < n2) && (xIndex < n1)  ){
        index_in = (zIndex * n2 + yIndex) * n1 + xIndex ; 
        block[threadIdx.y][threadIdx.x] = idata[index_in];
    }
    __syncthreads();

    yIndex = t2 * BLOCK_DIM + threadIdx.x ;
    xIndex = t1 * BLOCK_DIM + threadIdx.y ;
 	
    if ( (zIndex < n3) && (xIndex < n1) && (yIndex < n2)  ){
        index_out = (zIndex * n1 + xIndex) * n2 + yIndex ; 
        odata[index_out] = block[threadIdx.x][threadIdx.y] ;
    } 	
 
}
