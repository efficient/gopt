#include <stdio.h>
#include <stdarg.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <time.h>

#define USE_HUGEPAGE 1
#define USE_INTERLEAVING 1
#define INCLUDE_COPY_TIME 1

#define ITERS 100			/** < Number of measurements to average on */
#define DEPTH 1

#define MAX_PKTS (32768 * 128)

#define LOG_CAP (256 * 1024 * 1024)		// 256 M log
#define LOG_CAP_ ((256 * 1024 * 1024) - 1)	// 256 M log

#define CPE(val, msg, err_code) \
	if(val) { fprintf(stderr, msg); fprintf(stderr, " Error %d \n", err_code); \
	exit(err_code);}

void printDeviceProperties();
void red_printf(const char *format, ...);
