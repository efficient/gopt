#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<papi.h>
#include<time.h>

#include "fpp.h"
#include "ndn.h"

// #define URL_FILE "/home/akalia/fastpp/data_dump/ndn_distributed_sample"
#define URL_FILE "data/ndn_distributed_sample_small"

void process_batch(struct ndn_linear_url *url_lo) 
{
	int batch_index = 0;
	foreach(batch_index, BATCH_SIZE) {
	}
}

int main(int argc, char **argv)
{
	struct ndn_ht ht;
	int i;

	/** < Variables for PAPI */
	float real_time, proc_time, ipc;
	long long ins;
	int retval;

	red_printf("main: Initializing NDN hash table\n");
	ndn_init(URL_FILE, 0xf, &ht);
	red_printf("\tmain: Setting up NDN index done!\n");

	red_printf("main: Checking if all URLs were inserted\n");
	ndn_check(URL_FILE, &ht);
	red_printf("\tmain: Check succeeded\n");

	red_printf("main: Getting URL array\n");
	int nb_urls = ndn_get_num_urls(URL_FILE);
	struct ndn_linear_url *url_arr = ndn_get_url_array(URL_FILE);
	red_printf("\tmain: Constructed URL array!\n");

	red_printf("main: Starting NDN lookups\n");
	/** < Init PAPI_TOT_INS and PAPI_TOT_CYC counters */
	if((retval = PAPI_ipc(&real_time, &proc_time, &ins, &ipc)) < PAPI_OK) {    
		printf("PAPI error: retval: %d\n", retval);
		exit(1);
	}

	for(i = 0; i < nb_urls; i += BATCH_SIZE) {
		process_batch(&url_arr[i]);
	}

	if((retval = PAPI_ipc(&real_time, &proc_time, &ins, &ipc)) < PAPI_OK) {    
		printf("PAPI error: retval: %d\n", retval);
		exit(1);
	}

	red_printf("Time = %.4f s, rate = %.2f\n"
		"Instructions = %lld, IPC = %f\n", 
		real_time, nb_urls / real_time,
		ins, ipc);

	return 0;
}
