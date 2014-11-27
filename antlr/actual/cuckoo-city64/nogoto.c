#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<papi.h>
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
	foreach(batch_index, BATCH_SIZE) {
		int i, bkt_1, bkt_2, success = 0;
		int key = key_lo[batch_index];

		/** < Try the first bucket */
		bkt_1 = hash(key) & NUM_BKT_;
		FPP_EXPENSIVE(&ht_index[bkt_1]);
		
		for(i = 0; i < 8; i ++) {
			if(ht_index[bkt_1].slot[i].key == key) {
				sum += ht_index[bkt_1].slot[i].value;
				succ_1 ++;
				success = 1;
				break;
			}
		}

		if(success == 0) {
			bkt_2 = hash(bkt_1) & NUM_BKT_;
			FPP_EXPENSIVE(&ht_index[bkt_2]);
			
			for(i = 0; i < 8; i ++) {
				if(ht_index[bkt_2].slot[i].key == key) {
					sum += ht_index[bkt_2].slot[i].value;
					succ_2 ++;
					success = 1;
					break;
				}
			}
		}

		if(success == 0) {
			fail ++;
		}
	}
}

int main(int argc, char **argv)
{
	int i;

	/** < Variables for PAPI */
	float real_time, proc_time, ipc;
	long long ins;
	int retval;

	red_printf("main: Initializing cuckoo hash table\n");
	cuckoo_init(&keys, &ht_index);

	red_printf("main: Starting lookups\n");
	/** < Init PAPI_TOT_INS and PAPI_TOT_CYC counters */
	if((retval = PAPI_ipc(&real_time, &proc_time, &ins, &ipc)) < PAPI_OK) {    
		printf("PAPI error: retval: %d\n", retval);
		exit(1);
	}

	for(i = 0; i < NUM_KEYS; i += BATCH_SIZE) {
		process_batch(&keys[i]);
	}

	if((retval = PAPI_ipc(&real_time, &proc_time, &ins, &ipc)) < PAPI_OK) {    
		printf("PAPI error: retval: %d\n", retval);
		exit(1);
	}

	red_printf("Time = %.4f s, rate = %.2f\n"
		"Instructions = %lld, IPC = %f\n"		
		"sum = %d, succ_1 = %d, succ_2 = %d, fail = %d\n", 
		real_time, NUM_KEYS / real_time,
		ins, ipc,
		sum, succ_1, succ_2, fail);

	return 0;
}
