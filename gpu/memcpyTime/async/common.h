#include <stdio.h>
#include <stdarg.h>
#include <cuda_runtime.h>
#include <time.h>
#include <assert.h>

#define ITERS 1000

void printDeviceProperties();
void red_printf(const char *format, ...);

#define CPE(val, msg) \
	if(val) { fprintf(stderr, msg); \
	exit(-1);}
