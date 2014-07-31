#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<time.h>
#include<sys/ipc.h>
#include<sys/shm.h>

#include "fpp.h"
#include "param.h"

#define foreach(i, n) for(i = 0; i < n; i ++)

struct cache_bkt		/* 64 bytes */
{
	int slot_arr[SLOTS_PER_BKT];
};
struct cache_bkt *cache;

#define ABS(a) (a > 0 ? a : -1 * a)

// Each packet contains a random integer. The memory address accessed
// by the packet is determined by an expensive hash of the integer.
int *pkts;

int sum = 0;

// batch_index must be declared outside process_pkts_in_batch
int batch_index = 0;

// Process BATCH_SIZE pkts starting from lo
int process_pkts_in_batch(int *pkt_lo)
{
	int jumper[BATCH_SIZE];
	int iMask = 0, b_i;			// Completion mask and batch index

	// Phase 1: initialize jumper and issue 1st prefetech
	for(b_i = 0; b_i < BATCH_SIZE; b_i ++) {
		jumper[b_i] = pkt_lo[b_i];
		__builtin_prefetch(&cache[jumper[b_i]], 0, 0);
	}

	// Phase 2: Jump around a bit
	int i;
	for(i = 0; i < DEPTH; i++) {
		for(b_i = 0; b_i < BATCH_SIZE; b_i ++) {
			if(FPP_ISSET(iMask, b_i)) {
				continue;
			}
			int *arr = cache[jumper[b_i]].slot_arr;
			
			int j, best_j = 0;
			int max_diff = ABS(arr[0] - jumper[b_i]) % 8;

			for(j = 1; j < SLOTS_PER_BKT; j ++) {
				if(ABS(arr[j] - jumper[b_i]) % 8 > max_diff) {
					max_diff = ABS(arr[j] - jumper[b_i]) % 8;
					best_j = j;
				}
			}

			jumper[b_i] = arr[best_j];
			if(jumper[b_i] % 16 == 0) {
				iMask = FPP_SET(iMask, b_i);
			} else if (i != DEPTH - 1) {
				__builtin_prefetch(&cache[jumper[b_i]], 0, 0);
			}
		}
	}

	// Phase 3: accumulate
	for(b_i = 0; b_i < BATCH_SIZE; b_i ++) {
		sum += jumper[b_i];
	}

}

int main(int argc, char **argv)
{
	int i, j;

	// Allocate a large memory area
	fprintf(stderr, "Size of cache = %lu\n", NUM_BS * sizeof(struct cache_bkt));

	int sid = shmget(CACHE_SID, NUM_BS * sizeof(struct cache_bkt), 
		IPC_CREAT | 0666 | SHM_HUGETLB);
	if(sid < 0) {
		fprintf(stderr, "Could not create cache\n");
		exit(-1);
	}
	cache = shmat(sid, 0, 0);

	// Fill in the cache with index into itself
	for(i = 0; i < NUM_BS; i ++) {
		for(j = 0; j < SLOTS_PER_BKT; j++) {
			cache[i].slot_arr[j] = rand() & NUM_BS_;
		}
	}

	// Allocate the packets
	pkts = (int *) malloc(NUM_PKTS * sizeof(int));
	for(i = 0; i < NUM_PKTS; i++) {
		pkts[i] = rand() & NUM_BS_;
	}

	fprintf(stderr, "Finished creating cache and packets\n");

	long long start, end;
	start = get_cycles();

	for(i = 0; i < NUM_PKTS; i += BATCH_SIZE) {
		process_pkts_in_batch(&pkts[i]);
	}
	
	end = get_cycles();

	// xia-router2 frequency = 2.7 Ghz
	long long ns = ((long long) (end - start) / 2.7);

	printf("Total time = %f s, sum = %d\n", ns / 1000000000.0, sum);
	printf("Average time per batch = %lld ns\n", ns / (NUM_PKTS / BATCH_SIZE));
	printf("Average time per packet = %lld ns \n", ns / NUM_PKTS);

}
