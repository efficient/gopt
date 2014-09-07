#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <assert.h>

#define NUM_PKTS (1024 * 1024)		// For PCIe 1.0 x16 measurement
#define ITERS 10

void printDeviceProperties();
long long get_cycles();
cudaError_t checkCuda(cudaError_t result);

#define CPE(val, msg, err_code) \
	if(val) { fprintf(stderr, msg); fprintf(stderr, " Error %d \n", err_code); \
	exit(err_code);}
