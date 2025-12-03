
# global.h  main.cpp  transpose3D_cpu.cpp  transpose3D_revised.cu
# [lchien@cg transpose3D]$ nvcc -c -arch=sm_20 transpose3D_revised.cu 
#

LIBS = -L/usr/local/cuda/lib64 -lcudart 
INCPATH = -I/usr/local/cuda/include

all: global.h
	nvcc -c -arch=sm_20 transpose3D_revised.cu
	g++ -c transpose3D_cpu.cpp
	g++ -c $(INCPATH) main.cpp 
	g++ $(LIBS) transpose3D_revised.o transpose3D_cpu.o main.o

clean:
	rm -f *.o 
