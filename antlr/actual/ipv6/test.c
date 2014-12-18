#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "rte_lpm6.h"

#define IPV6_ADDR_SIZE 16

int main()
{
	int i, j;
	uint8_t ipv6_buf[IPV6_ADDR_SIZE] = {0};

	/**< Create the lmp6 struct */
	struct rte_lpm6_config ipv6_config;
	ipv6_config.max_rules = 1024;
	ipv6_config.number_tbl8s = 1024 * 1024;
	struct rte_lpm6 *lpm = rte_lpm6_create(0, &ipv6_config);

	/**< Read the prefixes from a prefixes file */
	FILE *prefix_fp = fopen("prefix_file", "r");
	assert(prefix_fp != NULL);
	int num_prefixes;
	fscanf(prefix_fp, "%d", &num_prefixes);
	assert(num_prefixes > 0);
	printf("Inserting %d prefixes\n", num_prefixes);

	for(i = 0; i < num_prefixes; i ++) {
		memset(ipv6_buf, 0, IPV6_ADDR_SIZE * sizeof(uint8_t));

		int prefix_depth, cur_byte, dst_port;

		fscanf(prefix_fp, "%d", &prefix_depth);

		for(j = 0; j < IPV6_ADDR_SIZE; j ++) {
			fscanf(prefix_fp, "%d", &cur_byte);
			assert(cur_byte >= 0 && cur_byte <= 255);

			ipv6_buf[j] = (uint8_t) cur_byte;
		}

		fscanf(prefix_fp, "%d", &dst_port);

		printf("prefix_depth = %d, dst_port = %d\n", prefix_depth, dst_port);
		
		rte_lpm6_add(lpm, ipv6_buf, prefix_depth, dst_port);
	}

	printf("\tDone inserting prefixes\n");
	

	/**< Read the probe IPv6 addresses from an IPs file */
	FILE *ips_fp = fopen("ip_file", "r");
	assert(ips_fp != NULL);
	int num_ips;
	fscanf(ips_fp, "%d", &num_ips);
	assert(num_ips > 0);
	printf("Probing %d ips\n", num_ips);

	for(i = 0; i < num_ips; i ++) {
		memset(ipv6_buf, 0, IPV6_ADDR_SIZE * sizeof(uint8_t));
		int cur_byte, dst_port;

		for(j = 0; j < IPV6_ADDR_SIZE; j ++) {
			fscanf(ips_fp, "%d", &cur_byte);
			assert(cur_byte >= 0 && cur_byte <= 255);

			ipv6_buf[j] = (uint8_t) cur_byte;
		}
		
		uint8_t next_hop;
		int success = rte_lpm6_lookup(lpm, ipv6_buf, &next_hop);
		int exp_dst_port;
		printf("IP #%d, success = %d, next_hop = %d\n",
			i, success, next_hop);
	}

	printf("\tDone probing IPs\n");

	return 0;

}
