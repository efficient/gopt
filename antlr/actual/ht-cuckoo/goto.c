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
	int val_sum = 0;

	int success[BATCH_SIZE];
	int bkt_2[BATCH_SIZE];
	int bkt_1[BATCH_SIZE];
	int i[BATCH_SIZE];
	int key[BATCH_SIZE];

	int I = 0;			// batch index
	void *batch_rips[BATCH_SIZE];		// goto targets
	int iMask = 0;		// No packet is done yet

	int temp_index;
	for(temp_index = 0; temp_index < BATCH_SIZE; temp_index ++) {
		batch_rips[temp_index] = &&fpp_start;
	}

fpp_start:

        success[I] = 0;
        key[I] = key_lo[I];
        
        /** < Try the first bucket */
        bkt_1[I] = hash(key[I]) & NUM_BKT_;
        FPP_PSS(&ht_index[bkt_1[I]], fpp_label_1);
fpp_label_1:

        for(i[I] = 0; i[I] < 8; i[I] ++) {
            if(ht_index[bkt_1[I]].slot[i[I]].key == key[I]) {
                val_sum += ht_index[bkt_1[I]].slot[i[I]].value;
                success[I] = 1;
                break;
            }
        }
        
        if(success[I] == 0) {
            bkt_2[I] = hash(bkt_1[I]) & NUM_BKT_;
            FPP_PSS(&ht_index[bkt_2[I]], fpp_label_2);
fpp_label_2:

            for(i[I] = 0; i[I] < 8; i[I] ++) {
                if(ht_index[bkt_2[I]].slot[i[I]].key == key[I]) {
					val_sum += ht_index[bkt_2[I]].slot[i[I]].value;
                    break;
                }
            }
        }
        
    
fpp_end:
    batch_rips[I] = &&fpp_end;
    iMask = FPP_SET(iMask, I); 
    if(iMask == (1 << BATCH_SIZE) - 1) {
        return val_sum;
    }
    I = (I + 1) & BATCH_SIZE_;
    goto *batch_rips[I];

	
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
