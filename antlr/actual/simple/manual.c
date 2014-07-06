#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<time.h>
#include<sys/ipc.h>
#include<sys/shm.h>

#include "param.h"

// Each packet contains a random integer. The memory address accessed
// by the packet is determined by an expensive hash of the integer.

int sum = 0;

int *ht_log;
#define LOG_CAP (128 * 1024 * 1024)
#define LOG_CAP_ ((128 * 1024 * 1024) - 1)
#define LOG_SID 1

int *pkts;
#define NUM_PKTS (32 * 1024 * 1024)

#define BATCH_SIZE 8
#define BATCH_SIZE_ 7

int batch_index = 0;

// Some compute function
// Increment 'a' by at most COMPUTE * 4: the return value is still random
int hash(int a)
{
	int ret = a;
	int i;
	for(i = 0; i < COMPUTE; i++) {
		ret = ret + ((i * ret) & 7);
	}

	return ret;
}

// Process BATCH_SIZE pkts starting from lo
int process_pkts_in_batch(int *pkt_lo)
{
	int mem_addr[BATCH_SIZE];
	// Like a foreach loop
	for(batch_index = 0; batch_index < BATCH_SIZE; batch_index ++) {
		mem_addr[batch_index] = hash(pkt_lo[batch_index]) & LOG_CAP_;
		__builtin_prefetch(&mem_addr[batch_index], 0, 0);
	}

	for(batch_index = 0; batch_index < BATCH_SIZE; batch_index ++) {
		sum += ht_log[mem_addr[batch_index]];
	}
}

int main(int argc, char **argv)
{
	int i;

	// Allocate a large memory area
	int sid = shmget(LOG_SID, LOG_CAP * sizeof(int), IPC_CREAT | 0666 | SHM_HUGETLB);
	if(sid < 0) {
		fprintf(stderr, "Could not create ht_log\n");
		exit(-1);
	}
	ht_log = shmat(sid, 0, 0);
	for(i = 0; i < LOG_CAP; i ++) {
		ht_log[i] = i;
	}

	// Allocate the packets
	pkts = (int *) malloc(NUM_PKTS * sizeof(int));
	for(i = 0; i < NUM_PKTS; i++) {
		pkts[i] = rand() & LOG_CAP_;
	}

	fprintf(stderr, "Finished creating ht_log and packets\n");

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
