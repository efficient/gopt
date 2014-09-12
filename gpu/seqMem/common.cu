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
