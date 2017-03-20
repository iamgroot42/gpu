CUCOMP := nvcc
CUFLAGS := --gpu-architecture=compute_35

pattern:
	${CUCOMP} pattern_GPU.cu -o pattern ${CUFLAGS}

clean:
	rm -rf pattern