#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<time.h>
#include<sys/ipc.h>
#include<sys/shm.h>

#include "param.h"

struct cache_bkt		/* 64 bytes */
{
	int slot_arr[SLOTS_PER_BKT];
};
struct cache_bkt *cache;

#define ISSET(n, i) (n & (1 << i))	// Is the ith bit of n == 1
#define SET(n, i) (n | (1 << i))	// Is the ith bit of n == 1

#define ABS(a) (a > 0 ? a : -1 * a)

// Prefetch, Save, and Switch
#define PSS(addr, label) \
do {\
	__builtin_prefetch(addr); \
	batch_rips[I] = &&label; \
	I = (I + 1) & BATCH_SIZE_;	\
	goto *batch_rips[I]; \
} while(0)
	 

// Each packet contains a random integer. The memory address accessed
// by the packet is determined by an expensive hash of the integer.
int *pkts;

int sum = 0;

// Process BATCH_SIZE pkts starting from lo
#include "fpp.h"
int process_pkts_in_batch(int *pkt_lo)
{
	int i[BATCH_SIZE];
	int jumper[BATCH_SIZE];
	int *arr[BATCH_SIZE];
	int best_j[BATCH_SIZE];
	int j[BATCH_SIZE];
	int max_diff[BATCH_SIZE];

	int I = 0;			// batch index
	void *batch_rips[BATCH_SIZE];		// goto targets
	int iMask = 0;		// No packet is done yet

	int temp_index;
	for(temp_index = 0; temp_index < BATCH_SIZE; temp_index ++) {
		batch_rips[temp_index] = &&label_0;
	}

label_0:

	// Like a foreach loop
	
		jumper[I] = pkt_lo[I];
			
		for(i[I] = 0; i[I] < DEPTH; i[I]++) {
			FPP_PSS(&cache[jumper[I]], label_1);
label_1:

			arr[I] = cache[jumper[I]].slot_arr;
			best_j[I] = 0;

			max_diff[I] = ABS(arr[I][0] - jumper[I]) % 8;

			for(j[I] = 1; j[I] < SLOTS_PER_BKT; j[I] ++) {
				if(ABS(arr[I][j[I]] - jumper[I]) % 8 > max_diff[I]) {
					max_diff[I] = ABS(arr[I][j[I]] - jumper[I]) % 8;
					best_j[I] = j[I];
				}
			}
			
			jumper[I] = arr[I][best_j[I]];
			if(jumper[I] % 16 == 0) {		// GCC will optimize this
				break;
			}
		}

		sum += jumper[I];
	
end:
    batch_rips[I] = &&end;
    iMask = FPP_SET(iMask, I); 
    if(iMask == (1 << BATCH_SIZE) - 1) {
        return;
    }
    I = (I + 1) & BATCH_SIZE_;
    goto *batch_rips[I];

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

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	for(i = 0; i < NUM_PKTS; i += BATCH_SIZE) {
		process_pkts_in_batch(&pkts[i]);
	}

	clock_gettime(CLOCK_REALTIME, &end);
	printf("Time = %f sum = %d\n", 
		(end.tv_sec - start.tv_sec) + (double) (end.tv_nsec - start.tv_nsec) / 1000000000,
		sum);
}
