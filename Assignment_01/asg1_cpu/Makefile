#OS X compilation flags
# CPPFLAGS=-Wno-deprecated-declarations -I. -I./devil/include
# LDFLAGS= -lstdc++ -O2 -L./devil/lib -lIL -lILU

#Linux compilation flags
CPPFLAGS=-Wno-deprecated-declarations -I./devil/include
LDFLAGS= -O2 -L./devil/lib -lm -lstdc++ -lIL -lILU

assignment02: 
	g++ -o edgeDetect edgeDetect_CPU.cpp ${CPPFLAGS} ${LDFLAGS}

clean:
	-rm -f edgeDetect