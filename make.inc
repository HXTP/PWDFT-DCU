#/usr/bin/bash

COMPILE_MODE     = release
#COMPILE_MODE     = debug
USE_COMPLEX      = 0
USE_OPENMP       = 0
USE_PROFILE      = 0
USE_PEXSI        = 0
USE_GPU          = 1

ifeq (${COMPILE_MODE}, release)
  COMPILE_DEF    = -DRELEASE
#  COMPILE_FLAG   = -fast -w -D GPUDIRECT -no-ipo
  COMPILE_FLAG   = -O3 -w 
endif
ifeq (${COMPILE_MODE}, debug)
  COMPILE_DEF    = -DDEBUG=1
  COMPILE_FLAG   = -O2 -w
endif

ifeq (${USE_COMPLEX}, 1)
  SCALAR_DEF    = -DCOMPLEXSCALAR
endif

HIP_DIR=/public/home/sghpc_sdk/Linux_x86_64/25.6/dtk/dcc-2506
LIBXC_DIR=/public/home/acents8f17/code/libxc-6.2.2
FFTW_DIR=/public/home/acents8f17/code/fftw-3.3.10
MKL_ROOT=/public/home/sghpc_sdk/Linux_x86_64/25.6/compilers/intel/2021.3.0/mkl
YAML_DIR=/public/home/acents8f17/code/yaml-cpp-0.8.0

ifeq (${USE_GPU}, 1)
  HIPCC = $(HIP_DIR)/hip/bin/hipcc
  HIP_LIB = -L$(HIP_DIR)/lib -lamdhip64 -lgalaxyhip -lhipblas -lrocfft -lhipfft -L$(HIP_DIR)/rocblas/lib -lrocblas  -L$(HIP_DIR)/rocsolver/lib -lrocsolver
  HIPCC_FLAG = -I$(HIP_DIR)/include -DGPU -I$(HIP_DIR)/hip/include -I$(HIP_DIR)/hip/include/hip/hcc_detail/cuda -I ../include --gpu-max-threads-per-block=1024
  HIP_FLAG = -D GPU -I$(HIP_DIR)/include -I$(HIP_DIR)/hip/include -I$(HIP_DIR)/hip/include/hip/hcc_detail/cuda -I ../include  --gpu-max-threads-per-block=1024
endif

ifeq (${USE_OPENMP}, 1)
  OPENMP_DEF   = -DOPENMP
	OPENMP_FLAG  = -fopenmp
endif

ifeq (${USE_PROFILE}, 1)
	PROFILE_FLAG  = -g -pg
endif

ifeq (${USE_PEXSI}, 1)
  PEXSI_DEF        = -DPEXSI

  PEXSI_DIR        = 
  DSUPERLU_DIR     = 
  METIS_DIR        = 
  SCOTCH_DIR       = 

  PEXSI_INCLUDE    = 
  PEXSI_SRC_LIB    = 
  DSUPERLU_LIB     = 
  SCOTCH_LIB       = 
  METIS_LIB        = 
  PEXSI_LIB        = 
  PEXSI_LIB        = 
endif

# Includes
DGDFT_INCLUDE    = -I$(DGDFT_DIR)/include
LIBXC_INCLUDE    = -I$(LIBXC_DIR)/include
FFTW_INCLUDE     = -I$(FFTW_DIR)/include
YAML_INCLUDE     = -I$(YAML_DIR)/include
#MFFT_INCLUDE     = -I$(MFFT_DIR)/include
INCLUDES     = ${DGDFT_INCLUDE} ${PEXSI_INCLUDE} ${LIBXC_INCLUDE} ${BLOPEX_INCLUDE} ${FFTW_INCLUDE} ${MFFT_INCLUDE} ${YAML_INCLUDE} ${HIPCC_INC}


# Libraries
#MFFT_LIB         = $(MFFT_DIR)/lib/libmfft.a
LIBXC_LIB        = $(LIBXC_DIR)/lib/libxc.a
FFTW_LIB         = -L${FFTW_DIR}/lib -lfftw3_mpi -lfftw3 -lm
YAML_LIB         = $(YAML_DIR)/lib64/libyaml-cpp.a
GFORTRAN_LIB = -lifcore -lifport -limf -lsvml -liomp5 -lirng
#GFORTRAN_LIB     = -lgfortran
DGDFT_LIB        = ${DGDFT_DIR}/src/libdgdft.a
RQRCP_LIB        = ${DGDFT_DIR}/external/rqrcp/librqrcp.a
LBFGS_LIB        = ${DGDFT_DIR}/external/lbfgs/liblbfgs.a
MKL_LIB          = -L${MKL_ROOT}/lib/intel64 -lmkl_core -lmkl_intel_lp64  -lmkl_sequential -lmkl_blacs_intelmpi_lp64  -lmkl_scalapack_lp64 -lpthread 

LIBS        = ${DGDFT_LIB} ${PEXSI_LIB} ${LIBXC_LIB} ${GFORTRAN_LIB} ${BLOPEX_LIB} ${LBFGS_LIB} ${RQRCP_LIB} ${MFFT_LIB} ${FFTW_LIB} ${MKL_LIB} ${YAML_LIB} ${IPM} ${HIP_LIB} --gpu-max-threads-per-block=1024

# compiler
CC	= mpiicc
CXX	= mpiicpc
FC	= ifort
LOADER	= mpiicpc

AR           = ar
ARFLAGS      = rvcu
# For System V based machine without ranlib, like Cray and SGI,
# use touch instead.
#RANLIB      = touch
RANLIB       = ranlib

RM           = rm
RMFLAGS      = -f

# Different compiling and linking options.
#


CFLAGS       = ${COMPILE_FLAG} ${OPENMP_FLAG} ${PROFILE_FLAG} ${INCLUDES} ${HIP_FLAG} --gpu-max-threads-per-block=1024
FFLAGS       = ${COMPILE_FLAG} ${OPENMP_FLAG} ${PROFILE_FLAG} ${INCLUDES} #${HIP_FLAG} --gpu-max-threads-per-block=1024
CXXFLAGS     = ${COMPILE_FLAG} ${OPENMP_FLAG} ${PROFILE_FLAG} ${INCLUDES} ${HIP_FLAG} -std=c++11 -fpermissive --gpu-max-threads-per-block=1024
CCDEFS       = ${COMPILE_DEF} ${PEXSI_DEF} ${SCALAR_DEF} ${OPENMP_DEF} --gpu-max-threads-per-block=1024
CPPDEFS      = ${COMPILE_DEF} ${PEXSI_DEF} ${SCALAR_DEF} ${OPENMP_DEF} --gpu-max-threads-per-block=1024
LOADOPTS     = ${OPENMP_FLAG} ${PROFILE_FLAG} ${LIBS} 


# Compilation replacement rules

%.o: %.c
	${CC} -c ${CFLAGS} ${CCDEFS} $< 
%.o: %.cpp
	${CXX} -c ${CXXFLAGS} ${CPPDEFS} $< 
%.o: %.f
	${FC} -c ${FFLAGS} $<
%.o: %.F
	${FC} -c ${FFLAGS} $<
%.o: %.f90
	${FC} -c ${FFLAGS} $<
%.o: %.F90
	${FC} -c ${FFLAGS} $<
%.o: %.cu
	$(HIPCC) -c -fPIC -fno-gpu-rdc --offload-arch=gfx936 --gpu-max-threads-per-block=1024  -fpermissive ${HIPCC_FLAG} $<

# Generate auto-dependencies (for cpp files now)
%.d: %.cpp
	@set -e; rm -f $@; \
	$(CXX) -M $(CXXFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@;\
	rm -f $@.$$$$
