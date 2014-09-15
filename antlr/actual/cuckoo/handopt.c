#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<time.h>

#include "fpp.h"
#include "cuckoo.h"

int *keys;
struct cuckoo_bkt *ht_index;

int sum = 0;
int succ_1 = 0;		/** < Number of lookups that succeed in bucket 1 */
int succ_2 = 0;		/** < Number of lookups that success in bucket 2 */
int fail = 0;		/** < Failed lookups */

// batch_index must be declared outside process_batch
int batch_index = 0;

void process_batch(int *key_lo) 
{
	int i, bkt_1[BATCH_SIZE], bkt_2[BATCH_SIZE], key[BATCH_SIZE];
	int success[BATCH_SIZE] = {0};
	
	/** < Issue prefetch for the 1st bucket*/
	for(batch_index = 0; batch_index < BATCH_SIZE; batch_index ++) {
		key[batch_index] = key_lo[batch_index];

		bkt_1[batch_index] = hash(key[batch_index]) & NUM_BKT_;
		__builtin_prefetch(&ht_index[bkt_1[batch_index]], 0, 0);
	}
		

	/** < Try the 1st bucket. If it fails, issue prefetch for bkt #2 */
	for(batch_index = 0; batch_index < BATCH_SIZE; batch_index ++) {

		for(i = 0; i < 8; i ++) {
			if(ht_index[bkt_1[batch_index]].slot[i].key == key[batch_index]) {
				sum += ht_index[bkt_1[batch_index]].slot[i].value;
				succ_1 ++;
				success[batch_index] = 1;
				break;
			}
		}
	
		if(success[batch_index] == 0) {
			bkt_2[batch_index] = hash(bkt_1[batch_index]) & NUM_BKT_;
			__builtin_prefetch(&ht_index[bkt_2[batch_index]], 0, 0);
		}
	}

	/** < For failed batch elements, try the 2nd bucket */
	for(batch_index = 0; batch_index < BATCH_SIZE; batch_index ++) {

		if(success[batch_index] == 0) {
			for(i = 0; i < 8; i ++) {
				if(ht_index[bkt_2[batch_index]].slot[i].key == key[batch_index]) {
					sum += ht_index[bkt_2[batch_index]].slot[i].value;
					succ_2 ++;
					success[batch_index] = 1;
					break;
				}
			}
			
			if(success[batch_index] == 0) {
				fail ++;
			}
		}
	}
}

int main(int argc, char **argv)
{
	int i;
	struct timespec start, end;
	double seconds;

	red_printf("main: Initializing cuckoo hash table\n");
	cuckoo_init(&keys, &ht_index);

	red_printf("main: Starting lookups\n");
	clock_gettime(CLOCK_REALTIME, &start);

	for(i = 0; i < NUM_KEYS; i += BATCH_SIZE) {
		process_batch(&keys[i]);
	}

	clock_gettime(CLOCK_REALTIME, &end);

	seconds = (end.tv_sec - start.tv_sec) + 
		(double) (end.tv_nsec - start.tv_nsec) / 1000000000;
	red_printf("Time = %.4f, rate = %.2f "
		"sum = %d, succ_1 = %d, succ_2 = %d, fail = %d\n", 
		seconds, NUM_KEYS / seconds, sum, succ_1, succ_2, fail);

	return 0;
}
