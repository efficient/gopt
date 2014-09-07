#include <stdio.h>
#include <pthread.h>
#include <time.h>
#include <assert.h>

#include <cuda_runtime.h>

#define NUM_PKTS 16		// 1 cacheline
#define ITERS 10000

void printDeviceProperties();
long long get_cycles();
cudaError_t checkCuda(cudaError_t result);

#define CPE(val, msg, err_code) \
	if(val) { fprintf(stderr, msg); fprintf(stderr, " Error %d \n", err_code); \
	exit(err_code);}
