# NOTE: This Makefile does NOT support auto-dependency for the .h files.
# If the header files are changed, do "make clean" first.

include ../../make.inc

SRCF90 = rqrcp.f90 check_qpf.f90  check_qrf.f90  partial_pdgeqpf.f90 \
			 partial_qr_swap.f90  
SRCF   = partial_pdgeqp3.f  partial_pdgeqrf.f  \
				 pdgeqp3.f  pdgeqp3s.f  pdqppiv.f

OBJS = ${SRCS:.cpp=.o} ${SRCF90:.f90=.o} ${SRCF:.f=.o}
DEPS = ${SRCS:.cpp=.d} ${SRCF90:.f90=.d} ${SRCF:.f=.d}

all: librqrcp.a

-include ${DEPS}

librqrcp.a: ${OBJS}
	$(AR) $(ARFLAGS) $@ ${OBJS}
	$(RANLIB) $@

cleanall:
	${RM} -f ${OBJS} ${DEPS} librqrcp.a *.d.*
