#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>

#define LOG_CAP (64 * 1024 * 1024)		// 512 MB
#define LOG_CAP_ (LOG_CAP - 1)

#define LOG_KEY 1

#define ITERS 10000

void printDeviceProperties();
long long get_cycles();
void waitForNonZero(volatile int *A, int N);

#define CPE(val, msg, err_code) \
	if(val) { fprintf(stderr, msg); fprintf(stderr, " Error %d \n", err_code); \
	exit(err_code);}
