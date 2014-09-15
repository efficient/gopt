#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<papi.h>
#include<time.h>
#include<sys/ipc.h>
#include<sys/shm.h>

#include "param.h"
#include "fpp.h"

int *ht_log;

// Each packet contains a random integer
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
	int i, j, retval;

	// Variables for PAPI
	float real_time, proc_time, ipc;
	long long ins;

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

	// Init PAPI_TOT_INS and PAPI_TOT_CYC counters
	if((retval = PAPI_ipc(&real_time, &proc_time, &ins, &ipc)) < PAPI_OK) {    
		printf("retval: %d\n", retval);
		exit(1);
	}
	
	for(i = 0; i < NUM_PKTS; i += BATCH_SIZE) {
		process_pkts_in_batch(&pkts[i]);
	}

	if((retval = PAPI_ipc(&real_time, &proc_time, &ins, &ipc)) < PAPI_OK) {    
		printf("retval: %d\n", retval);
		exit(1);
	}

	printf("Sum = %d\n", sum);
	red_printf("Real_time: %.4fs, rate = %.2f\n"
		"Total instructions: %lld, Total cycles = %lld, IPC: %f\n", 
		real_time, NUM_PKTS / real_time,
		ins, (long long ) (ins / ipc), ipc);
}
