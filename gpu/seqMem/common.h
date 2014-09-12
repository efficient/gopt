#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define USE_HUGEPAGE 1

#define CUDA_THREADS (1536 * 32)		// GTX 690 has 1536 cores
#define LOG_CAP (192 * 1024 * 1024)		// 256 M log
#define LOG_CAP_ ((192 * 1024 * 1024) - 1)	// 256 M log

void printDeviceProperties();

#define CPE(val, msg, err_code) \
	if(val) { fprintf(stderr, msg); fprintf(stderr, " Error %d \n", err_code); \
	exit(err_code);}
