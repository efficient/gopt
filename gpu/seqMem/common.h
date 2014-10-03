#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define USE_HUGEPAGE 1
#define WR_ONLY 1

#define CUDA_THREADS (2048 * 32)		// GTX 980 has 2048 cores
#define LOG_CAP (256 * 1024 * 1024)		// 2 GB
#define LOG_CAP_ ((256 * 1024 * 1024) - 1)

void printDeviceProperties();

#define CPE(val, msg, err_code) \
	if(val) { fprintf(stderr, msg); fprintf(stderr, " Error %d \n", err_code); \
	exit(err_code);}
