#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<time.h>
#include<sys/ipc.h>
#include<sys/shm.h>

#include "param.h"

#define foreach(i, n) for(i = 0; i < n; i ++)


int sum = 0;

int *ht_log;
#define LOG_CAP (128 * 1024 * 1024)
#define LOG_CAP_ ((128 * 1024 * 1024) - 1)
#define LOG_SID 1

// Each packet contains a random integer. The memory address accessed
// by the packet is determined by an expensive hash of the integer.
int *pkts;
#define NUM_PKTS (16 * 1024 * 1024)

#define BATCH_SIZE 8
#define BATCH_SIZE_ 7


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

// batch_index must be declared outside process_pkts_in_batch
int batch_index = 0;

// Process BATCH_SIZE pkts starting from lo
int process_pkts_in_batch(int *pkt_lo)
{
	// Like a foreach loop
	foreach(batch_index, BATCH_SIZE) {

		int a_1 = hash(pkt_lo[batch_index]) & LOG_CAP_;
		int a_2 = hash(a_1) & LOG_CAP_;
		int a_3 = hash(a_2) & LOG_CAP_;
		int a_4 = hash(a_3) & LOG_CAP_;
		int a_5 = hash(a_4) & LOG_CAP_;
		int a_6 = hash(a_5) & LOG_CAP_;
		int a_7 = hash(a_6) & LOG_CAP_;
		int a_8 = hash(a_7) & LOG_CAP_;
		int a_9 = hash(a_8) & LOG_CAP_;
		int a_10 = hash(a_9) & LOG_CAP_;
		int a_11 = hash(a_10) & LOG_CAP_;
		int a_12 = hash(a_11) & LOG_CAP_;
		int a_13 = hash(a_12) & LOG_CAP_;
		int a_14 = hash(a_13) & LOG_CAP_;
		int a_15 = hash(a_14) & LOG_CAP_;
		int a_16 = hash(a_15) & LOG_CAP_;
		int a_17 = hash(a_16) & LOG_CAP_;
		int a_18 = hash(a_17) & LOG_CAP_;
		int a_19 = hash(a_18) & LOG_CAP_;
		int a_20 = hash(a_19) & LOG_CAP_;
		
		sum += ht_log[a_20];
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
