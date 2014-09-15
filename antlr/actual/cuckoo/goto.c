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
                sum += ht_index[bkt_1[I]].slot[i[I]].value;
                succ_1 ++;
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
                    sum += ht_index[bkt_2[I]].slot[i[I]].value;
                    succ_2 ++;
                    success[I] = 1;
                    break;
                }
            }
        }
        
        if(success[I] == 0) {
            fail ++;
        }
    
fpp_end:
    batch_rips[I] = &&fpp_end;
    iMask = FPP_SET(iMask, I); 
    if(iMask == (1 << BATCH_SIZE) - 1) {
        return;
    }
    I = (I + 1) & BATCH_SIZE_;
    goto *batch_rips[I];

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
