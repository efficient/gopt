#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<time.h>

#include "city.h"
#include "param.h"

// City hash of an unsigned number
uint32_t cityhash(uint32_t u)
{
	return CityHash32((char *) &u, 4);
}

int main(int argc, char **argv)
{
	printf("Starting computing hashes\n");
	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	int sum = 0, i = 0;
	for(i = 0; i < NUM_PKTS; i ++) {
		sum += CityHash32((char *) &i, 4);
	}

	clock_gettime(CLOCK_REALTIME, &end);
	double seconds = (end.tv_sec - start.tv_sec) + 
		(double) (end.tv_nsec - start.tv_nsec) / 1000000000;
	double nanoseconds = (end.tv_sec - start.tv_sec) * 1000000000 + 
		(end.tv_nsec - start.tv_nsec);
	printf("Time = %f, time per hash = %f ns, sum = %d\n", seconds, nanoseconds / NUM_PKTS, sum);
}
