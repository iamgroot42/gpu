
CPPCOMP := g++
CPPFLAGS := -Wno-deprecated-declarations

# CUCOMP := nvcc
# CUFLAGS := --gpu-architecture=compute_35

pattern:
	${CPPCOMP} pattern_CPU.cpp -o pattern ${CPPFLAGS}

clean:
	rm -rf pattern
