#include <stdio.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>

#include <cuda_runtime.h>

#define LOG_CAP 8192		// 4 KB
#define LOG_STEP 16

#define ITERS 1000

void printDeviceProperties();
long long get_cycles();
cudaError_t checkCuda(cudaError_t result);
void waitForNonZero(volatile int *A, int N);

#define CPE(val, msg, err_code) \
	if(val) { fprintf(stderr, msg); fprintf(stderr, " Error %d \n", err_code); \
	exit(err_code);}
