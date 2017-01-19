GPPFLAGS=-I./devil/include
LDFLAGS= -O2 -lm -lstdc++ -lIL -lILU

global: 
	nvcc -o edgeDetect_GPU edgeDetect_GPU.cu ${GPPFLAGS} ${LDFLAGS}

texture: 
	nvcc -o edgeDetect_GPU_texture edgeDetect_GPU_texture.cu ${GPPFLAGS} ${LDFLAGS}

clean:
	-rm -f edgeDetect_GPU edgeDetect_GPU_texture
