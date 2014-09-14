#include <stdio.h>
#include <stdarg.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>

#include <cuda_runtime.h>

#define ITERS 100000

void printDeviceProperties();
void waitForNonZero(volatile int *A, int N);
double get_timespec_us(struct timespec start, struct timespec end);
int cmpfunc (const void *a, const void *b);
void red_printf(const char *format, ...);

#define CPE(val, msg) \
	if(val) { fprintf(stderr, msg); \
	exit(-1);}
