#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<time.h>

#include "param.h"
#include "murmur3.h"

int main(int argc, char **argv)
{
	printf("Starting computing hashes\n");
	int seed = 42;
	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	int sum = 0, i = 0, res;
	for(i = 0; i < NUM_PKTS; i ++) {
		MurmurHash3_x86_32((char *) &i, 4, seed, &res);
		sum += res;
	}

	clock_gettime(CLOCK_REALTIME, &end);
	double seconds = (end.tv_sec - start.tv_sec) + 
		(double) (end.tv_nsec - start.tv_nsec) / 1000000000;
	double nanoseconds = (end.tv_sec - start.tv_sec) * 1000000000 + 
		(end.tv_nsec - start.tv_nsec);
	printf("Time = %f, time per hash = %f ns, sum = %d\n", seconds, nanoseconds / NUM_PKTS, sum);
}
