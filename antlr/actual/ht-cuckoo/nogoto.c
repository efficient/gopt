#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<time.h>
#include<unistd.h>

#include "fpp.h"
#include "cuckoo.h"

int *keys;
struct cuckoo_bkt *ht_index;

int process_batch(int *key_lo) 
{
	int batch_index = 0;
	int val_sum = 0;	/**< Cumulative value for all lookups that succeeded */

	foreach(batch_index, BATCH_SIZE) {
		int i, bkt_1, bkt_2, success = 0;
		int key = key_lo[batch_index];

		/** < Try the first bucket */
		bkt_1 = hash(key) & NUM_BKT_;
		FPP_EXPENSIVE(&ht_index[bkt_1]);
		
		for(i = 0; i < 8; i ++) {
			if(ht_index[bkt_1].slot[i].key == key) {
				val_sum += ht_index[bkt_1].slot[i].value;
				success = 1;
				break;
			}
		}

		if(success == 0) {
			bkt_2 = hash(bkt_1) & NUM_BKT_;
			FPP_EXPENSIVE(&ht_index[bkt_2]);
			
			for(i = 0; i < 8; i ++) {
				if(ht_index[bkt_2].slot[i].key == key) {
					val_sum += ht_index[bkt_2].slot[i].value;
					break;
				}
			}
		}
	}

	return val_sum;
}

void *cuckoo_thread(void *arg)
{
	int i;
	int id = *((int *) (arg));
	int tot_val_sum = 0;
	red_printf("Thread %d: Starting lookups\n", id);

	struct timespec start, end;

	while(1) {
		clock_gettime(CLOCK_REALTIME, &start);

		for(i = 0; i < NUM_KEYS; i += BATCH_SIZE) {
			tot_val_sum += process_batch(&keys[i]);
		}

		clock_gettime(CLOCK_REALTIME, &end);
		double seconds = (end.tv_sec - start.tv_sec) +
			(double) (end.tv_nsec - start.tv_nsec) / 1000000000;

		red_printf("Thread ID: %d, Rate = %.2f M/s. Value sum = %d\n",
			id, NUM_KEYS / (seconds * 1000000), tot_val_sum);
		
	}

}

int main(int argc, char **argv)
{
	int i;

	assert(argc == 2);
	int num_threads = atoi(argv[1]);
	assert(num_threads >= 1 && num_threads <= CUCKOO_MAX_THREADS);

	red_printf("main: Initializing shared cuckoo hash table\n");
	cuckoo_init(&keys, &ht_index);

	/**< Thread structures */
	pthread_t worker_threads[CUCKOO_MAX_THREADS];

	for(i = 0; i < num_threads; i ++) {
		int tid = i;
		pthread_create(&worker_threads[i], NULL, cuckoo_thread, &tid);

		/**< Ensure that threads don't use the same keys close in time */
		sleep(1);
	}

	for(i = 0; i < num_threads; i ++) {
		pthread_join(worker_threads[i], NULL);
	}

	/**< The work never ends */
	assert(0);

	return 0;
}
