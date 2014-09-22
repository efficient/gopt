#include <sys/ipc.h>
#include <sys/shm.h>
#include "common.h"

int main(int argc, char **argv)
{
	int i;
	/** < num_pkts is passed as a command-line flag */
	assert(argc == 2);
	int num_pkts = atoi(argv[1]);

	/** < Initialize hugepage log */
	int *log, *pkts, sum;

	int sid = shmget(LOG_KEY, LOG_CAP * sizeof(int), 
		SHM_HUGETLB | 0666 | IPC_CREAT);
	assert(sid >= 0);
	log = (int *) shmat(sid, 0, 0);
	assert(log != NULL);

	// Log data should be non-zero
	for(i = 0; i < LOG_CAP; i ++) {
		log[i] = i + 1;	
	}

	/** < Allocate addresses */
	pkts = (int *) malloc(num_pkts * sizeof(int));

	/** < We can only get full execution measurements */
	struct timespec start, end;
	double tot = 0, iter_us;

	int iter = 0, j = 0;

	for(iter = 0; iter < ITERS; iter ++) {
		/** < Write input addresses into A */
		for(i = 0; i < num_pkts; i ++) {
			pkts[i] = rand() & LOG_CAP_;
		}

		/** < Start a timer */
		clock_gettime(CLOCK_REALTIME, &start);

		/** < Do the expensive memory accesses */
		for(j = 0; j < num_pkts; j ++) {
			sum += log[pkts[j]];
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

	free(pkts);

	/** < Remove shared memory */
	shmdt(log);
	shmctl(sid, IPC_RMID, 0);

	printf("Done\n");
	return 0;
}

