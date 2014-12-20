#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "rte_lpm.h"

#define IPV4_ADDR_SIZE 4

#define MAX_IPV4_RULES (2 * 1024 * 1024)
#define PREFIX_FILE "java/prefix_file"
#define IP_FILE "java/ip_file"

int main()
{
	int i, j;
	uint32_t ipv4_buf[IPV4_ADDR_SIZE] = {0};

	/**< Create the lmp struct on socket 0 */
	struct rte_lpm *lpm = rte_lpm_create(0, MAX_IPV4_RULES);

	/**< Read the prefixes from a prefixes file */
	FILE *prefix_fp = fopen(PREFIX_FILE, "r");
	assert(prefix_fp != NULL);
	int num_prefixes;
	fscanf(prefix_fp, "%d", &num_prefixes);
	assert(num_prefixes > 0);
	printf("Inserting %d prefixes\n", num_prefixes);

	for(i = 0; i < num_prefixes; i ++) {
		memset(ipv4_buf, 0, IPV4_ADDR_SIZE * sizeof(uint8_t));

		int prefix_depth, cur_byte, dst_port;
		uint32_t prefix_ip = 0;

		fscanf(prefix_fp, "%d", &prefix_depth);

		for(j = 0; j < IPV4_ADDR_SIZE; j ++) {
			fscanf(prefix_fp, "%d", &cur_byte);
			assert(cur_byte >= 0 && cur_byte <= 255);

			ipv4_buf[j] = cur_byte;
			prefix_ip += (ipv4_buf[j] << (8 * (3 - j)));
		}

		fscanf(prefix_fp, "%d", &dst_port);
		printf("prefix_depth = %d, dst_port = %d, prefix_ip = %x "
			"bytes= %x %x %x %x\n",
			prefix_depth, dst_port, prefix_ip,
			ipv4_buf[0], ipv4_buf[1], ipv4_buf[2], ipv4_buf[3]);
		
		int add_status = rte_lpm_add(lpm, prefix_ip, prefix_depth, dst_port);
		if(add_status < 0) {
			printf("Failed to add IPv4 prefix %d. Status = %d\n",
				i, add_status);
			exit(-1);
		}
	}

	printf("\tDone inserting prefixes\n");
	
	/**< Read the probe IPv4 addresses from an IPs file */
	FILE *ips_fp = fopen(IP_FILE, "r");
	assert(ips_fp != NULL);
	int num_ips;
	fscanf(ips_fp, "%d", &num_ips);
	assert(num_ips > 0);
	printf("Probing %d ips\n", num_ips);

	int *dst_ports = malloc(num_ips * sizeof(int));

	for(i = 0; i < num_ips; i ++) {
		memset(ipv4_buf, 0, IPV4_ADDR_SIZE * sizeof(uint8_t));
		int cur_byte;
		uint32_t probe_ip = 0;

		for(j = 0; j < IPV4_ADDR_SIZE; j ++) {
			fscanf(ips_fp, "%d", &cur_byte);
			assert(cur_byte >= 0 && cur_byte <= 255);

			ipv4_buf[j] = (uint8_t) cur_byte;

			probe_ip += (ipv4_buf[j] << (8 * (3 - j)));
		}
		
		rte_lpm_lookup(lpm, probe_ip, (uint8_t *) &dst_ports[i]);
	}

	/**< Check if the computed dst ports are correct */
	for(i = 0; i < num_ips; i ++) {
		int exp_dst_port;
		fscanf(ips_fp, "%d", &exp_dst_port);
		if(dst_ports[i] == exp_dst_port) {
			printf("IP %d passed! Got: %d, Expected: %d\n",
				i, dst_ports[i], exp_dst_port);
		} else {
			printf("IP %d failed! Got: %d, Expected: %d\n",
				i, dst_ports[i], exp_dst_port);
			exit(-1);
		}
	}

	printf("\tDone probing IPs\n");

	return 0;

}
