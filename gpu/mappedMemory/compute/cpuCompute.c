#include "city.h"
#include "common.h"

int main(int argc, char **argv)
{
	/** < num_pkts is passed as a command-line flag */
	assert(argc == 2);
	int num_pkts = atoi(argv[1]);
	int sum = 0;

	int *A = (int *) malloc(num_pkts * sizeof(int));
	
	/** < We can only get full execution measurements */
	struct timespec start, end;
	double tot, iter_us;

	int iter = 0, j = 0;

	for(iter = 0; iter < ITERS; iter ++) {

		/** < Start a timer */
		clock_gettime(CLOCK_REALTIME, &start);
		
		/** < Write input data into A */
		for(j = 0; j < num_pkts; j ++) {
			A[j] = j + iter + 1;
		}

		for(j = 0; j < num_pkts; j ++) {
			sum += CityHash32((char *) &A[j], 4);
		}
		/** < Stop timer */
		clock_gettime(CLOCK_REALTIME, &end);

		iter_us = (end.tv_sec - start.tv_sec) * 1000000 + 
			(double) (end.tv_nsec - start.tv_nsec) / 1000;

		printf("\tIter %d: %.2f us. Per packet: %.2f ns, sum = %d\n", 
			iter, iter_us, iter_us * 1000 / num_pkts, sum);

		tot += iter_us;
	}

	printf("Average %.2f us\n", tot / ITERS);

	free(A);

	printf("Done\n");
	return 0;
}

