#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define USE_HUGEPAGE 0
#define USE_INTERLEAVING 0
#define DEPTH 20

#define NUM_PKTS (8 * 1024 * 1024)
#define LOG_CAP (64 * 1024 * 1024)		// 256 M log
#define LOG_CAP_ ((64 * 1024 * 1024) - 1)	// 256 M log

void printDeviceProperties();

#define CPE(val, msg, err_code) \
	if(val) { fprintf(stderr, msg); fprintf(stderr, " Error %d \n", err_code); \
	exit(err_code);}
