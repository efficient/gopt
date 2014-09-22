#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdarg.h>
#include <cuda_runtime.h>
#include <time.h>

#define COMPUTE 20

void printDeviceProperties();
void red_printf(const char *format, ...);

#define CPE(val, msg, err_code) \
	if(val) { fprintf(stderr, msg); fprintf(stderr, " Error %d \n", err_code); \
	exit(err_code);}
