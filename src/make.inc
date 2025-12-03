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

ifeq (${USE_GPU}, 1)
  HIP_DIR = /public/software/compiler/dtk/dtk-23.04
  HIPCC = $(HIP_DIR)/hip/bin/hipcc
  HIP_LIB = -L$(HIP_DIR)/lib -lamdhip64 -lgalaxyhip -lhipblas -lrocfft -lhipfft -L$(HIP_DIR)/rocblas/lib -lrocblas  -L$(HIP_DIR)/rocsolver/lib -lrocsolver #-L/public/software/mathlib/magma/magma-rocm_3.3_develop/lib -lmagma 
  HIPCC_FLAG = -I$(HIP_DIR)/include -DGPU -I$(HIP_DIR)/hip/include -I$(HIP_DIR)/hip/include/hip/hcc_detail/cuda -I ../include #-I/public/software/mathlib/magma/magma-rocm_3.3_develop/include
  HIP_FLAG = -D GPU -I$(HIP_DIR)/include -I$(HIP_DIR)/hip/include -I$(HIP_DIR)/hip/include/hip/hcc_detail/cuda -I ../include #-I/public/software/mathlib/magma/magma-rocm_3.3_develop/include
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
DGDFT_INCLUDE    = -I${DGDFT_DIR}/include
LIBXC_INCLUDE    = -I$(LIBXC_DIR)/include
LIBXC_INCLUDE    = -I$(LIBXC_DIR)/include
FFTW_INCLUDE     = -I${FFTW_DIR}/include
YAML_INCLUDE     = -I$(YAML_DIR)/include

INCLUDES     = ${DGDFT_INCLUDE} ${PEXSI_INCLUDE} ${LIBXC_INCLUDE} ${BLOPEX_INCLUDE} ${FFTW_INCLUDE} ${YAML_INCLUDE} ${HIPCC_INC}


# Libraries

LIBXC_LIB        = $(LIBXC_DIR)/lib/libxc.a
FFTW_LIB         = -L${FFTW_DIR}/lib -lfftw3_mpi -lfftw3 -lm
YAML_LIB         = $(YAML_DIR)/build/libyaml-cpp.a
GFORTRAN_LIB     = -lgfortran
DGDFT_LIB        = ${DGDFT_DIR}/src/libdgdft.a
RQRCP_LIB        = ${DGDFT_DIR}/external/rqrcp/librqrcp.a
LBFGS_LIB        = ${DGDFT_DIR}/external/lbfgs/liblbfgs.a
MKL_LIB          = -L${MKL_ROOT}/lib/intel64 -lmkl_core -lmkl_intel_lp64  -lmkl_sequential -lmkl_blacs_openmpi_lp64  -lmkl_scalapack_lp64 -lpthread -lgfortran

LIBS        = ${DGDFT_LIB} ${PEXSI_LIB} ${LIBXC_LIB} ${GFORTRAN_LIB} ${BLOPEX_LIB} ${LBFGS_LIB} ${RQRCP_LIB} ${FFTW_LIB} ${MKL_LIB} ${YAML_LIB} ${IPM} ${HIP_LIB} 

# compiler
CC	= mpicc
CXX	= mpic++
FC	= mpif90
LOADER	= mpic++

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


CFLAGS       = ${COMPILE_FLAG} ${OPENMP_FLAG} ${PROFILE_FLAG} ${INCLUDES} ${HIP_FLAG}
FFLAGS       = ${COMPILE_FLAG} ${OPENMP_FLAG} ${PROFILE_FLAG} ${INCLUDES} #${HIP_FLAG}
CXXFLAGS     = ${COMPILE_FLAG} ${OPENMP_FLAG} ${PROFILE_FLAG} ${INCLUDES} ${HIP_FLAG} -std=c++11 -fpermissive
CCDEFS       = ${COMPILE_DEF} ${PEXSI_DEF} ${SCALAR_DEF} ${OPENMP_DEF}
CPPDEFS      = ${COMPILE_DEF} ${PEXSI_DEF} ${SCALAR_DEF} ${OPENMP_DEF}
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
	$(HIPCC) -c -fPIC -fno-gpu-rdc --amdgpu-target=gfx906 -fpermissive ${HIPCC_FLAG} $<

# Generate auto-dependencies (for cpp files now)
%.d: %.cpp
	@set -e; rm -f $@; \
	$(CXX) -M $(CXXFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@;\
	rm -f $@.$$$$
