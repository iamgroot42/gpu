#Linux compilation flags
#CPPFLAGS=-Wno-deprecated-declarations -I./devil/include
#LDFLAGS= -O2 -L./devil/lib -lm -lstdc++ -lIL -lILU

#assignment02: 
#	g++ -o edgeDetect edgeDetect_CPU.cpp ${CPPFLAGS} ${LDFLAGS}

#clean:
#	-rm -f edgeDetect


GPPFLAGS=-I./devil/include -Wno-deprecated-gpu-targets
LDFLAGS= -O2 -lm -lstdc++ -lIL -lILU

assignment02: 
	nvcc -o ohho a.cu ${GPPFLAGS} ${LDFLAGS}

clean:
	-rm -f ohho

