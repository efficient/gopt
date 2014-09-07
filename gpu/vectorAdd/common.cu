#include "common.h"

void printDeviceProperties()
{
	struct cudaDeviceProp deviceProp;
	int ret = cudaGetDeviceProperties(&deviceProp, 0);
	CPE(ret != cudaSuccess, "Get Device Properties failed\n", -1);

	printf("Device name: %s\n", deviceProp.name);
	printf("Total global memory: %lu bytes\n", deviceProp.totalGlobalMem);
	printf("Warp size: %d\n", deviceProp.warpSize);
	printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);

	printf("Multi-processor count: %d\n", deviceProp.multiProcessorCount);
	printf("Threads per multi-processor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
}
