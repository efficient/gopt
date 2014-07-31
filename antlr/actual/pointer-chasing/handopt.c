#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<time.h>
#include<sys/ipc.h>
#include<sys/shm.h>

#include "param.h"
#include "fpp.h"

int *ht_log;

// Each packet contains a random integer. The memory address accessed
// by the packet is determined by an expensive hash of the integer.
int *pkts;

int sum = 0;

// batch_index must be declared outside process_pkts_in_batch
int batch_index = 0;

// Process BATCH_SIZE pkts starting from lo
int process_pkts_in_batch(int *pkt_lo)
{
	// Stage 1: issue prefetches
	int jumper[BATCH_SIZE];
	for(batch_index = 0; batch_index < BATCH_SIZE; batch_index ++) {
		jumper[batch_index] = pkt_lo[batch_index];
		__builtin_prefetch(&ht_log[jumper[batch_index]], 0, 0);
	}

	// Stage 2: jump around
	int i;
	for(i = 0; i < DEPTH; i++) {
		for(batch_index = 0; batch_index < BATCH_SIZE; batch_index ++) {
			jumper[batch_index] = ht_log[jumper[batch_index]];
			if(i != DEPTH - 1) {
				__builtin_prefetch(&ht_log[jumper[batch_index]], 0, 0);
			}
		}
	}

	// Stage 3: accumulate
	for(batch_index = 0; batch_index < BATCH_SIZE; batch_index ++) {
		sum += jumper[batch_index];
	}
}

int main(int argc, char **argv)
{
	int i, j;

	// Allocate a large memory area
	fprintf(stderr, "Size of ht_log = %lu\n", LOG_CAP * sizeof(int));

	int sid = shmget(LOG_SID, LOG_CAP * sizeof(int), IPC_CREAT | 0666 | SHM_HUGETLB);
	if(sid < 0) {
		fprintf(stderr, "Could not create ht_log\n");
		exit(-1);
	}
	ht_log = shmat(sid, 0, 0);

	// Fill in the ht_log with index into itself
	for(i = 0; i < LOG_CAP; i ++) {
		ht_log[i] = rand() & LOG_CAP_;
	}

	// Allocate the packets
	pkts = (int *) malloc(NUM_PKTS * sizeof(int));
	for(i = 0; i < NUM_PKTS; i++) {
		pkts[i] = rand() & LOG_CAP_;
	}

	fprintf(stderr, "Finished creating ht_log and packets\n");

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
