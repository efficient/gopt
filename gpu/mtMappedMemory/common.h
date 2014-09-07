#include <stdio.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>

#include <cuda_runtime.h>

#define CACHELINE_CAP 16			// Capacity of a cacheline, in ints
#define LOG_CAP (1024 * 1024)		// 4 MB
#define ITERS 1000

void printDeviceProperties();
long long get_cycles();
cudaError_t checkCuda(cudaError_t result);
void waitForNonZero(volatile int *A, int N, int tid);

#define CPE(val, msg, err_code) \
	if(val) { fprintf(stderr, msg); fprintf(stderr, " Error %d \n", err_code); \
	exit(err_code);}
