# NOTE: This Makefile does NOT support auto-dependency for the .h files.
# If the header files are changed, do "make clean" first.

include ../make.inc

SRCS = pwdft.cpp dgdft.cpp \
	diagonalize.cpp den2cube.cpp alb2cube.cpp lufact.cpp den2cubepw.cpp 

OBJS = ${SRCS:.cpp=.o}
DEPS = ${SRCS:.cpp=.d}
EXES = ${SRCS:.cpp=}	

pwdft: pwdft.o ${DGDFT_LIB} 
	($(LOADER) -o $@ pwdft.o $(LOADOPTS) )
#pwdft: pwdft.o cuda_utils.o ${DGDFT_LIB} 
#	($(LOADER) -o $@ pwdft.o cuda_utils.o $(LOADOPTS) )

dgdft: dgdft.o ${DGDFT_LIB} 
	($(LOADER) -o $@ dgdft.o $(LOADOPTS) )

diagonalize: diagonalize.o ${DGDFT_LIB} 
	($(LOADER) -o $@ diagonalize.o $(LOADOPTS) )

den2cubepw: den2cubepw.o ${DGDFT_LIB} 
	($(LOADER) -o $@ den2cubepw.o $(LOADOPTS) )

den2cube: den2cube.o ${DGDFT_LIB} 
	($(LOADER) -o $@ den2cube.o $(LOADOPTS) )

alb2cube: alb2cube.o ${DGDFT_LIB} 
	($(LOADER) -o $@ alb2cube.o $(LOADOPTS) )

lufact: lufact.o ${DGDFT_LIB} 
	($(LOADER) -o $@ lufact.o $(LOADOPTS) )

-include ${DEPS}

${DGDFT_LIB}:
	(cd ${DGDFT_DIR}/src; make all)

cleanlib:
	(${RM} -f ${DGDFT_LIB})

cleanall:
	rm -f ${EXES} ${OBJS} ${DEPS} *.d.o
