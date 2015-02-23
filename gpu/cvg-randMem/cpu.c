#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <pthread.h>

#include "common.h"

void *cpu_func(void *ptr)
{
	int i, j;
	long long iter = 0;
	int tid = ((struct thread_info *) ptr)->tid;
	int *log = ((struct thread_info *) ptr)->log;
	int *pkts;
	struct timespec start, end;

	printf("Running tid %d\n", tid);

	/**< Initialize a local packet array */
	pkts = (int *) malloc(NUM_PKTS * sizeof(int));
	for(i = 0; i < NUM_PKTS; i ++) {
		pkts[i] = rand() & LOG_CAP_;
	}
	
	while(1) {
		iter ++;
		
		clock_gettime(CLOCK_REALTIME, &start);

		/**< Follow 8 chains simultaneously */
		for(i = 0; i < NUM_PKTS; i += 8) {
			for(j = 0; j < DEPTH; j ++) {
				pkts[i] = log[pkts[i]];
				pkts[i + 1] = log[pkts[i + 1]];
				pkts[i + 2] = log[pkts[i + 2]];
				pkts[i + 3] = log[pkts[i + 3]];
				pkts[i + 4] = log[pkts[i + 4]];
				pkts[i + 5] = log[pkts[i + 5]];
				pkts[i + 6] = log[pkts[i + 6]];
				pkts[i + 7] = log[pkts[i + 7]];
			}
		}

		/**< Prevent chains from getting into cycles */
		for(i = 0; i < NUM_PKTS; i ++) {
			pkts[i] = (pkts[i] + i) & LOG_CAP_;
		}

		clock_gettime(CLOCK_REALTIME, &end);
		double time = (double) (end.tv_nsec - start.tv_nsec) / 1000000000 + 
			(end.tv_sec - start.tv_sec);

		if((iter & 1023) == 0) {
			/**< Print the number of packets processed per second */
			printf("num_pkts %d TID %d %.2f, sample = %d\n", NUM_PKTS, tid,
				NUM_PKTS / (time * 1000000), pkts[iter % NUM_PKTS]);
		}
	}
}

int main(int argc, char *argv[])
{
	pthread_t thread[MAX_THREADS];
	int i, num_threads;
	int *log;

	assert(argc == 2);
	num_threads = atoi(argv[1]);
	assert(num_threads >= 1 && num_threads <= MAX_THREADS);

	/**< The method of dependent pointer chasing used here can lead to cycles
	  *  (that improve cache hit rate) with very large depth. */
	assert(DEPTH <= 10);
	assert(NUM_PKTS % 8 == 0);

	/**< Initialize hugepage log for all CPU threads */
	red_printf("Allocating host log of size %lu bytes\n", LOG_CAP * sizeof(int));

	int sid = shmget(LOG_KEY,
		LOG_CAP * sizeof(int), SHM_HUGETLB | 0666 | IPC_CREAT);
	assert(sid >= 0);
	log = (int *) shmat(sid, 0, 0);
	assert(log != NULL);

	for(i = 0; i < LOG_CAP; i ++) {
		log[i] = rand() % LOG_CAP;
	}

	/**< Start all CPU threads */
	for(i = 0; i < num_threads; i ++) {
		struct thread_info ti;
		ti.tid = i;
		ti.log = log;
		pthread_create(&thread[i], NULL, cpu_func, (void *) &ti);

		/**< Allow threads to go out of sync */
		usleep(100000);
	}

	/**< Wait till the sun rises in the west and sets in the east */
	for(i = 0; i < num_threads; i ++) {
		pthread_join(thread[i], NULL);
	}

}

