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

// Process BATCH_SIZE pkts starting from lo
int process_pkts_in_batch(int *pkt_lo)
{
	int I = 0;
	void *batch_rips[BATCH_SIZE];
	int a_1[BATCH_SIZE];
	int a_2[BATCH_SIZE];
	int a_3[BATCH_SIZE];
	int a_4[BATCH_SIZE];
	int a_5[BATCH_SIZE];
	int a_6[BATCH_SIZE];
	int a_7[BATCH_SIZE];
	int a_8[BATCH_SIZE];
	int a_9[BATCH_SIZE];
	int a_10[BATCH_SIZE];
	int a_11[BATCH_SIZE];
	int a_12[BATCH_SIZE];
	int a_13[BATCH_SIZE];
	int a_14[BATCH_SIZE];
	int a_15[BATCH_SIZE];
	int a_16[BATCH_SIZE];
	int a_17[BATCH_SIZE];
	int a_18[BATCH_SIZE];
	int a_19[BATCH_SIZE];
	int a_20[BATCH_SIZE];

	int temp_index;
	for(temp_index = 0; temp_index < BATCH_SIZE; temp_index ++) {
		batch_rips[temp_index] = &&label_0;
	}

label_0:
	
	a_1[I] = hash(pkt_lo[I]) & LOG_CAP_;
	a_2[I] = hash(a_1[I]) & LOG_CAP_;
	a_3[I] = hash(a_2[I]) & LOG_CAP_;
	a_4[I] = hash(a_3[I]) & LOG_CAP_;
	a_5[I] = hash(a_4[I]) & LOG_CAP_;
	a_6[I] = hash(a_5[I]) & LOG_CAP_;
	a_7[I] = hash(a_6[I]) & LOG_CAP_;
	a_8[I] = hash(a_7[I]) & LOG_CAP_;
	a_9[I] = hash(a_8[I]) & LOG_CAP_;
	a_10[I] = hash(a_9[I]) & LOG_CAP_;
	a_11[I] = hash(a_10[I]) & LOG_CAP_;
	a_12[I] = hash(a_11[I]) & LOG_CAP_;
	a_13[I] = hash(a_12[I]) & LOG_CAP_;
	a_14[I] = hash(a_13[I]) & LOG_CAP_;
	a_15[I] = hash(a_14[I]) & LOG_CAP_;
	a_16[I] = hash(a_15[I]) & LOG_CAP_;
	a_17[I] = hash(a_16[I]) & LOG_CAP_;
	a_18[I] = hash(a_17[I]) & LOG_CAP_;
	a_19[I] = hash(a_18[I]) & LOG_CAP_;
	a_20[I] = hash(a_19[I]) & LOG_CAP_;
	
	__builtin_prefetch(&ht_log[a_10[I]], 0, 0);
	batch_rips[I] = &&label_1;

	I = (I + 1) & BATCH_SIZE_;
	if(I != 0) {
		goto *batch_rips[I];
	}

label_1:
	sum += ht_log[a_10[I]];
	I = (I + 1) & BATCH_SIZE_;
	if(I != 0) {
		goto *batch_rips[I];
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
