#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<time.h>

#include "city.h"

#define NUM_PKTS (16 * 1024 * 1024)
#define NUM_LONGS 4

int main(int argc, char **argv)
{
	printf("Starting computing hashes\n");
	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	int sum = 0, i = 0, j = 0;
	long long A[NUM_LONGS];

	for(i = 0; i < NUM_PKTS; i ++) {
		for(j = 0; j < NUM_LONGS; j ++) {
			A[j] = 0xffffffffffffffffL + i;
		}
		sum += CityHash32((char *) A, NUM_LONGS * sizeof(long long));
	}

	clock_gettime(CLOCK_REALTIME, &end);
	double seconds = (end.tv_sec - start.tv_sec) + 
		(double) (end.tv_nsec - start.tv_nsec) / 1000000000;
	double nanoseconds = (end.tv_sec - start.tv_sec) * 1000000000 + 
		(end.tv_nsec - start.tv_nsec);
	printf("Time = %f, time per hash = %f ns, sum = %d\n", seconds, nanoseconds / NUM_PKTS, sum);

	return 0;
}
