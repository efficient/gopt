#include <cuda_runtime.h>
#include "common.h"

// Like printf, but red. Limited to 1000 characters.
void red_printf(const char *format, ...)
{	
	#define RED_LIM 1000
	va_list args;
	int i;

	char buf1[RED_LIM], buf2[RED_LIM];
	memset(buf1, 0, RED_LIM);
	memset(buf2, 0, RED_LIM);

    va_start(args, format);

	// Marshal the stuff to print in a buffer
	vsnprintf(buf1, RED_LIM, format, args);

	// Probably a bad check for buffer overflow
	for(i = RED_LIM - 1; i >= RED_LIM - 50; i --) {
		assert(buf1[i] == 0);
	}

	// Add markers for red color and reset color
	snprintf(buf2, 1000, "\033[31m%s\033[0m", buf1);

	// Probably another bad check for buffer overflow
	for(i = RED_LIM - 1; i >= RED_LIM - 50; i --) {
		assert(buf2[i] == 0);
	}

	printf("%s", buf2);

    va_end(args);
}

void printDeviceProperties()
{
	struct cudaDeviceProp deviceProp;
	int ret = cudaGetDeviceProperties(&deviceProp, 0);
	CPE(ret != cudaSuccess, "Get Device Properties failed\n");

	printf("\n=================DEVICE PROPERTIES=================\n");
	printf("\tDevice name: %s\n", deviceProp.name);
	printf("\tTotal global memory: %lu bytes\n", deviceProp.totalGlobalMem);
	printf("\tWarp size: %d\n", deviceProp.warpSize);
	printf("\tCompute capability: %d.%d\n", deviceProp.major, deviceProp.minor);

	printf("\tMulti-processor count: %d\n", deviceProp.multiProcessorCount);
	printf("\tThreads per multi-processor: %d\n", deviceProp.maxThreadsPerMultiProcessor);

	printf("\n");
}

// Returns when all N elements in A are non-zero
void waitForNonZero(volatile int *A, int N)
{
	int i, turns = 0;
	while(1) {
		int allNonZero = 1;
		for(i = 0; i < N; i ++) {
			if(A[i] == 0) {
				allNonZero = 0;
			}
		}
		if(allNonZero) {
			return;
		}

		turns ++;
		if(turns > 1000000) {
			printf("Waiting for non-zero ...\n");
			turns = 0;
		}
	}
}

/** < Useful for sorting an array of doubles */
int cmpfunc (const void *a, const void *b)
{
	double a_d = *(double *) a;
	double b_d = *(double *) b;

	if(a_d > b_d) {
		return 1;
	} else if(a_d < b_d) {
		return -1;
	} else {
		return 0;
	}
}

double get_timespec_us(struct timespec start, struct timespec end)
{
	double ret = 
		(double) (end.tv_nsec - start.tv_nsec) / 1000 +
		(end.tv_sec - start.tv_sec) * 1000000;

	return ret;
}
