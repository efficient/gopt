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
	int i[BATCH_SIZE];
	int jumper[BATCH_SIZE];

	int I = 0;			// batch index
	void *batch_rips[BATCH_SIZE];		// goto targets
	int iMask = 0;		// No packet is done yet

	int temp_index;
	for(temp_index = 0; temp_index < BATCH_SIZE; temp_index ++) {
		batch_rips[temp_index] = &&fpp_start;
	}

fpp_start:

    // Like a foreach loop
    
        jumper[I] = pkt_lo[I];
        
        for(i[I] = 0; i[I] < DEPTH; i[I]++) {
            FPP_PSS(&ht_log[jumper[I]], fpp_label_1);
fpp_label_1:

            jumper[I] = ht_log[jumper[I]];
        }
        
        sum += jumper[I];
       
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
