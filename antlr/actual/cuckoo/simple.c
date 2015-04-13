#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<papi.h>
#include<time.h>

#include "fpp.h"
#include "cuckoo.h"

struct cuckoo_bkt *ht_index;

int process_batch(int key)
{
	int i, bkt_1, bkt_2, ret = -1;

	/**< Try the first bucket */
	bkt_1 = hash(key) & NUM_BKT_;
		
	for(i = 0; i < 8; i ++) {
		if(ht_index[bkt_1].slot[i].key == key) {
			ret = ht_index[bkt_1].slot[i].value;
			break;
		}
	}

	if(ret == -1) {
		bkt_2 = hash(bkt_1) & NUM_BKT_;
		
		for(i = 0; i < 8; i ++) {
			if(ht_index[bkt_2].slot[i].key == key) {
				ret = ht_index[bkt_2].slot[i].value;
				break;
			}
		}
	}

	return ret;
}
