#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>

#include "rte_lpm.h"
#include "ipv4.h"

#define IPV4_ADDR_SIZE 4

#define MAX_IPV4_RULES (2 * 1024 * 1024)
#define PREFIX_FILE "../../../data_dump/ipv4/ipv4_java_out"
#define IP_FILE "java/ip_file"

#define NUM_ADDRS (64 * 1024 * 1024)

int main()
{
	int i, j;

	/**< Create the lmp struct on socket 0 */
	struct rte_lpm *lpm = rte_lpm_create(0, MAX_IPV4_RULES);

	/**< Read the prefixes from a prefixes file */
	int num_prefixes;
	struct ipv4_prefix *prefix_arr = ipv4_read_prefixes(PREFIX_FILE,
		&num_prefixes);
		
	for(i = 0; i < num_prefixes; i ++) {
		uint32_t prefix_ip = 0;

		for(j = 0; j < IPV4_ADDR_SIZE; j ++) {
			prefix_ip += (prefix_arr[i].bytes[j] << (8 * (3 - j)));
		}

		int add_status = rte_lpm_add(lpm,
			prefix_ip, prefix_arr[i].depth, prefix_arr[i].dst_port);
		if(add_status < 0) {
			printf("main: Failed to add IPv4 prefix %d. Status = %d\n",
				i, add_status);
			exit(-1);
		}

		if(i % 1000 == 0) {
			printf("main: Added %d IPs\n", i);
		}
	}

	printf("\tmain: Done inserting prefixes\n");
	
	/**< Generate the probe IPv4 addresses from prefixes */
	printf("\tmain: Generating IPv4 addresses\n");
	struct ipv4_addr *addr_arr = ipv4_gen_addrs(NUM_ADDRS,
		prefix_arr, num_prefixes);
		
	uint8_t *dst_ports = malloc(NUM_ADDRS * sizeof(uint8_t));
	int dst_port_sum = 0;

	for(i = 0; i < NUM_ADDRS; i ++) {
		uint32_t probe_ip = 0;

		for(j = 0; j < IPV4_ADDR_SIZE; j ++) {
			probe_ip += (addr_arr[i].bytes[j] << (8 * (3 - j)));
		}
		
		rte_lpm_lookup(lpm, probe_ip, (uint8_t *) &dst_ports[i]);
		ipv4_print_addr(&addr_arr[i]);
		printf("\t%d\n", dst_ports[i]);

		usleep(200000);
		dst_port_sum += dst_ports[i];
	}

	printf("\tDone probing IPs. dst_port_sum = %d\n", dst_port_sum);

	return 0;

}
