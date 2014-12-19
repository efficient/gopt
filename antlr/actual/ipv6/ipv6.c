#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "rte_lpm6.h"
#include "ipv6.h"

struct ipv6_prefix *ipv6_read_prefixes(const char *prefixes_file,
	int *num_prefixes)
{
	assert(prefixes_file != NULL && num_prefixes != NULL);

	FILE *prefix_fp = fopen(prefixes_file, "r");
	assert(prefix_fp != NULL);

	fscanf(prefix_fp, "%d", num_prefixes);
	assert(*num_prefixes > 0);
	printf("ipv6: Reading %d prefixes\n", *num_prefixes);

	int prefix_mem_size = *num_prefixes * sizeof(struct ipv6_prefix);
	struct ipv6_prefix *prefix_arr = malloc(prefix_mem_size);
	assert(prefix_arr != NULL);

	int i, j;
	for(i = 0; i < *num_prefixes; i ++) {
		/**< A prefix is formatted as <depth> <bytes 0 ... 15> <dst port> */
		fscanf(prefix_fp, "%d", &prefix_arr[i].depth);

		for(j = 0; j < IPV6_ADDR_LEN; j ++) {
			int new_byte;
			fscanf(prefix_fp, "%d", &new_byte);
			assert(new_byte >= 0 && new_byte <= 255);

			prefix_arr[i].bytes[j] = new_byte;
		}

		fscanf(prefix_fp, "%d", &prefix_arr[i].dst_port);
	}

	return prefix_arr;	
}

struct ipv6_addr *ipv6_gen_addrs(int num_addrs,
	struct ipv6_prefix *prefix_arr, int num_prefixes)
{
	assert(num_addrs > 0 && prefix_arr != NULL && num_prefixes > 0);

	struct ipv6_addr *addr_arr;
	int addr_mem_size = num_addrs * sizeof(struct ipv6_addr);

	addr_arr = hrd_malloc_socket(PROBE_ADDR_SHM_KEY, addr_mem_size, 0);

	/**< Generate addresses using randomly chosen prefixes */
	int i;
	for(i = 0; i < num_addrs; i ++) {
		int prefix_id = rand() % num_prefixes;
		memcpy(addr_arr[i].bytes, prefix_arr[prefix_id].bytes, IPV6_ADDR_LEN);
	}

	return addr_arr;
}
