#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <papi.h>
#include <time.h>
#include "rte_lpm6.h"
#include "ipv6.h"

#define PREFIX_FILE "../../../data_dump/ipv6/ipv6_java_out"
#define NUM_IPS (64 * 1024 * 1024)
#define AMP_FACTOR 2
#define BATCH_SIZE 8

int main()
{
	int i, j;

	/**< Create the lmp6 struct */
	struct rte_lpm6_config ipv6_config;
	ipv6_config.max_rules = 1000000;
	ipv6_config.number_tbl8s = 1024 * 1024;
	struct rte_lpm6 *lpm = rte_lpm6_create(0, &ipv6_config);

	/**< Read the prefixes from a prefixes file */
	struct ipv6_prefix *prefix_arr;
	int num_prefixes;

	prefix_arr = ipv6_read_prefixes(PREFIX_FILE, &num_prefixes);
	printf("main: Read %d prefixes. Amplifying by %d.\n",
		num_prefixes, AMP_FACTOR);

	prefix_arr = ipv6_amp_prefixes(prefix_arr, num_prefixes, AMP_FACTOR);
	num_prefixes *= AMP_FACTOR;
	assert(num_prefixes < ipv6_config.max_rules);

	for(i = 0; i < num_prefixes; i ++) {
		int add_status = rte_lpm6_add(lpm,
			prefix_arr[i].bytes, prefix_arr[i].depth, prefix_arr[i].dst_port);

		if(add_status < 0) {
			printf("main: Failed to add IPv6 prefix %d. Status = %d\n",
				i, add_status);
			exit(-1);
		}

		if(i % 1000 == 0) {
			printf("main: Added prefixes = %d, total = %d\n", i, num_prefixes);
		}
	}

	printf("\tmain: Done inserting prefixes\n");
	
	/**< Generate probe IPv6 addresses from inserted prefixes */
	printf("main: Generating %d IPv6 addresses\n", NUM_IPS);
	struct ipv6_addr *addr_arr = ipv6_gen_addrs(NUM_IPS,
		prefix_arr, num_prefixes);

	printf("main: Starting lookups\n");

	/**< Variables for PAPI */
	float real_time, proc_time, ipc;
	long long ins;
	int retval;

	/**< Init PAPI_TOT_INS and PAPI_TOT_CYC counters */
	if((retval = PAPI_ipc(&real_time, &proc_time, &ins, &ipc)) < PAPI_OK) {
		printf("PAPI error: retval: %d\n", retval);
		exit(1);
	}

	int16_t dst_port[BATCH_SIZE];
	int dst_port_sum = 0;
	for(i = 0; i < NUM_IPS; i += BATCH_SIZE) {
		rte_lpm6_lookup_bulk_func(lpm, (void *) addr_arr[i].bytes, dst_port, BATCH_SIZE);
		for(j = 0; j < BATCH_SIZE; j ++) {
			dst_port_sum += dst_port[j];
		}
	}

	if((retval = PAPI_ipc(&real_time, &proc_time, &ins, &ipc)) < PAPI_OK) {
		printf("PAPI error: retval: %d\n", retval);
		exit(1);
	}

	printf("Time = %.4f s, Lookup rate = %.2f M/s | dst_port_sum = %d\n"
		"Instructions = %lld, IPC = %f\n",
		real_time, NUM_IPS / (real_time * 1000000), dst_port_sum, ins, ipc);

	return 0;

}
