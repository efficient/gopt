#include "common.h"

void printDeviceProperties()
{
	struct cudaDeviceProp deviceProp;
	int ret = cudaGetDeviceProperties(&deviceProp, 0);
	CPE(ret != cudaSuccess, "Get Device Properties failed\n", -1);

	printf("\n=================DEVICE PROPERTIES=================\n");
	printf("\tDevice name: %s\n", deviceProp.name);
	printf("\tTotal global memory: %lu bytes\n", deviceProp.totalGlobalMem);
	printf("\tWarp size: %d\n", deviceProp.warpSize);
	printf("\tCompute capability: %d.%d\n", deviceProp.major, deviceProp.minor);

	printf("\tMulti-processor count: %d\n", deviceProp.multiProcessorCount);
	printf("\tThreads per multi-processor: %d\n", deviceProp.maxThreadsPerMultiProcessor);

	printf("\n");
}

long long get_cycles()
{
	unsigned low, high;
	unsigned long long val;
	asm volatile ("rdtsc" : "=a" (low), "=d" (high));
	val = high;
	val = (val << 32) | low;
	return val;
}

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call.
cudaError_t checkCuda(cudaError_t result)
{
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %sn", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
	return result;
}
