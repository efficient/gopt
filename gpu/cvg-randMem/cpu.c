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
	int *streams;
	struct timespec start, end;

	printf("Running tid %d\n", tid);

	/**< Initialize the streams */
	streams = (int *) malloc(CPU_NUM_STREAMS * sizeof(int));
	for(i = 0; i < CPU_NUM_STREAMS; i ++) {
		streams[i] = rand() & LOG_CAP_;
	}
	
	clock_gettime(CLOCK_REALTIME, &start);

	while(1) {
		iter ++;

		/**< Follow 8 chains simultaneously */
		for(i = 0; i < CPU_NUM_STREAMS; i += 8) {
			for(j = 0; j < DEPTH; j ++) {
				streams[i] = log[streams[i]];
				streams[i + 1] = log[streams[i + 1]];
				streams[i + 2] = log[streams[i + 2]];
				streams[i + 3] = log[streams[i + 3]];
				streams[i + 4] = log[streams[i + 4]];
				streams[i + 5] = log[streams[i + 5]];
				streams[i + 6] = log[streams[i + 6]];
				streams[i + 7] = log[streams[i + 7]];
			}
		}

		if(iter == CPU_ITERS) {
			clock_gettime(CLOCK_REALTIME, &end);
			double time = (double) (end.tv_nsec - start.tv_nsec) / 1000000000 + 
				(end.tv_sec - start.tv_sec);

			/**< Print the number of packets processed per second */
			printf("TID %d (%d streams) %.2f, sample = %d\n",
				tid, CPU_NUM_STREAMS,
				(CPU_NUM_STREAMS * CPU_ITERS) / (time * 1000000),
				streams[iter % CPU_NUM_STREAMS]);
			
			clock_gettime(CLOCK_REALTIME, &start);

			iter = 0;
		}
	}
}

int main(int argc, char *argv[])
{
	pthread_t thread[CPU_MAX_THREADS];
	int i, num_threads;
	int *log;

	assert(argc == 2);
	num_threads = atoi(argv[1]);
	assert(num_threads >= 1 && num_threads <= CPU_MAX_THREADS);

	/**< We follow 8 streams in one shot */
	assert(CPU_NUM_STREAMS % 8 == 0);

	/**< Initialize hugepage log for all CPU threads */
	red_printf("Allocating host log of size %lu bytes\n", LOG_CAP * sizeof(int));

	int sid = shmget(LOG_KEY,
		LOG_CAP * sizeof(int), SHM_HUGETLB | 0666 | IPC_CREAT);
	assert(sid >= 0);
	log = (int *) shmat(sid, 0, 0);
	assert(log != NULL);

	init_ht_log(log, LOG_CAP);

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

