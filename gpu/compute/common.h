#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>


void printDeviceProperties();
long long get_cycles(void);

#define CPE(val, msg, err_code) \
	if(val) { fprintf(stderr, msg); fprintf(stderr, " Error %d \n", err_code); \
	exit(err_code);}
