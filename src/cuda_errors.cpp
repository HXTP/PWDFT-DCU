#ifdef GPU
#include "cuda_errors.hpp"
#include <hipblas/hipblas.h>
char *cublasGetErrorString(hipblasStatus_t error)
{
    switch (error)
    {
        case HIPBLAS_STATUS_SUCCESS:
            return "HIPBLAS_STATUS_SUCCESS";

        case HIPBLAS_STATUS_NOT_INITIALIZED:
            return "HIPBLAS_STATUS_NOT_INITIALIZED";

        case HIPBLAS_STATUS_ALLOC_FAILED:
            return "HIPBLAS_STATUS_ALLOC_FAILED";

        case HIPBLAS_STATUS_INVALID_VALUE:
            return "HIPBLAS_STATUS_INVALID_VALUE";

        case HIPBLAS_STATUS_ARCH_MISMATCH:
            return "HIPBLAS_STATUS_ARCH_MISMATCH";

        case HIPBLAS_STATUS_MAPPING_ERROR:
            return "HIPBLAS_STATUS_MAPPING_ERROR";

        case HIPBLAS_STATUS_EXECUTION_FAILED:
            return "HIPBLAS_STATUS_EXECUTION_FAILED";

        case HIPBLAS_STATUS_INTERNAL_ERROR:
            return "HIPBLAS_STATUS_INTERNAL_ERROR";

        default:
            return "<unknown>";

/*        case rocblas_status_success:
            return "rocblas_status_success";

        case rocblas_status_invalid_handle:
            return "rocblas_status_invalid_handle";

        case rocblas_status_not_implemented:
            return "rocblas_status_not_implemented";

        case rocblas_status_invalid_pointer:
            return "rocblas_status_invalid_pointer";

        case rocblas_status_invalid_size:
            return "rocblas_status_invalid_size";

        case rocblas_status_memory_error:
            return "rocblas_status_memory_error";

        case rocblas_status_internal_error:
            return "rocblas_status_internal_error";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
*/
    }
}

// returns string for CUFFT API error
char *cufftGetErrorString(rocfft_status error)
{
    switch (error)
    {
	case rocfft_status_success:
            return "rocfft_status_success";

        case rocfft_status_failure:
            return "rocfft_status_failure";

        case rocfft_status_invalid_arg_value:
            return "rocfft_status_invalid_arg_value";

        case rocfft_status_invalid_dimensions:
            return "rocfft_status_invalid_dimensions";

        case rocfft_status_invalid_array_type:
            return "rocfft_status_invalid_array_type";

        case rocfft_status_invalid_strides:
            return "rocfft_status_invalid_strides";

        case rocfft_status_invalid_distance:
            return "rocfft_status_invalid_distance";

        case rocfft_status_invalid_offset:
            return "rocfft_status_invalid_offset";
/*
        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
*/
	default:
    	    return "<unknown>";
    }
}

#endif
