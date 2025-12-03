# NOTE: This Makefile does NOT support auto-dependency for the .h files.
# If the header files are changed, do "make clean" first.

include ../make.inc

SRCS = scf.cpp eigensolver.cpp  iondynamics.cpp \
       hamiltonian_dg_matrix.cpp hamiltonian_dg.cpp hamiltonian.cpp \
       periodtable.cpp spinor.cpp fourier.cpp utility.cpp esdf.cpp \
       mpi_interf.cpp scalapack.cpp lapack.cpp blas.cpp global.cpp \
       tddft.cpp  sgmres.cpp chebyshev_dg.cpp mixing_dg.cpp \
       scf_dg_setup.cpp  scf_dg_inner.cpp  scf_dg_outer.cpp scf_dg_diag.cpp \
       hamiltonian_dg_hfx.cpp hamiltonian_gpu.cpp eigensolver_gpu.cpp \
       spinor_gpu.cpp cuda_errors.cpp  cublas.cpp cusolver.cpp

#magma.cpp
SRCLRTDDFT = lrtddft.cpp

#SRCF90 = rqrcp.f90 check_qpf.f90  check_qrf.f90  partial_pdgeqpf.f90 \
#			 partial_qr_swap.f90  
#SRCF   = partial_pdgeqp3.f  partial_pdgeqrf.f  \
#				 pdgeqp3.f  pdgeqp3s.f  pdqppiv.f
SRCCU  = cuda_utils.cu

ifeq (${USE_GPU}, 1)
ifeq (${USE_COMPLEX}, 1) 
OBJS = ${SRCS:.cpp=.o} ${SRCF90:.f90=.o} ${SRCF:.f=.o} ${SRCCU:.cu=.o}
else
OBJS = ${SRCS:.cpp=.o} ${SRCLRTDDFT:.cpp=.o} ${SRCF90:.f90=.o} ${SRCF:.f=.o} ${SRCCU:.cu=.o}
endif
else
ifeq (${USE_COMPLEX}, 1)
OBJS = ${SRCS:.cpp=.o} ${SRCF90:.f90=.o} ${SRCF:.f=.o} 
else
OBJS = ${SRCS:.cpp=.o} ${SRCLRTDDFT:.cpp=.o} ${SRCF90:.f90=.o} ${SRCF:.f=.o}
endif
endif
DEPS = ${SRCS:.cpp=.d} ${SRCF90:.f90=.d} ${SRCF:.f=.d}

all: libdgdft.a

-include ${DEPS}

libdgdft.a: ${OBJS}
	$(AR) $(ARFLAGS) $@ ${OBJS}
	$(RANLIB) $@

cleanall:
	${RM} -f ${OBJS} ${DEPS} libdgdft.a *.d.*
