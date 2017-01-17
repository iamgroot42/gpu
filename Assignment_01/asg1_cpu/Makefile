GPPFLAGS=-I./devil/include
LDFLAGS= -O2 -lm -lstdc++ -lIL -lILU

global: 
	nvcc -o ohho a.cu ${GPPFLAGS} ${LDFLAGS}

texture: 
	nvcc -o ohho b.cu ${GPPFLAGS} ${LDFLAGS}

clean:
	-rm -f ohho
