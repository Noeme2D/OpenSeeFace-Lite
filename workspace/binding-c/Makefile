CC=gcc -std=c99 -Werror -Wall -Wextra -Wno-deprecated -Wno-unused-parameter
CYTHON=cython -X language_level=3 -X wraparound=False -X boundscheck=False -X cdivision=True

CFLAGS=$(shell python-config --includes)
CFLAGS+=-I$(shell python -c "import numpy; print(numpy.get_include())")
CFLAGS+=-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

LDFLAGS=$(shell python-config --ldflags --embed)

test: test.c osflite.c Makefile
	$(CC) $(CFLAGS) test.c osflite.c $(LDFLAGS) -o test

osflite.c: osflite.pyx
	$(CYTHON) osflite.pyx

clean:
	rm -f osflite.c
	rm -f osflite.h
	rm -f test