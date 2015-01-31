#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "rte_lpm6.h"
#include "ipv6.h"

/**< Read IPv6 prefixes from a file */
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

/**< Generate IPv6 prefixes randomly */
struct ipv6_prefix *ipv6_gen_rand_prefixes(int num_prefixes)
{
	assert(num_prefixes > 0);

	int prefix_mem_size = num_prefixes * sizeof(struct ipv6_prefix);
	struct ipv6_prefix *prefix_arr = malloc(prefix_mem_size);
	assert(prefix_arr != NULL);

	int i, j;
	for(i = 0; i < num_prefixes; i ++) {

		/**< Prefix depth is between 48 and 64 */
		prefix_arr[i].depth = (rand() % 17) + 48;

		for(j = 0; j < IPV6_ADDR_LEN; j ++) {
			prefix_arr[i].bytes[j] = rand() % 256;
		}

		prefix_arr[i].dst_port = rand() % 256;
	}

	return prefix_arr;
}

/**< Increase the number of prefixes in prefix_arr */
struct ipv6_prefix *ipv6_amp_prefixes(struct ipv6_prefix *prefix_arr,
	int num_prefixes, int amp_factor)
{
	int mem_size = num_prefixes * amp_factor * sizeof(struct ipv6_prefix);
	struct ipv6_prefix *new_prefix_arr = malloc(mem_size);
	assert(new_prefix_arr != NULL);

	struct ipv6_perm *perm_arr = ipv6_gen_perms(amp_factor);

	int i, j, k;
	for(i = 0; i < num_prefixes * amp_factor; i += amp_factor) {

		/**< New prefixes i, ..., i + amp_factor - 1 come from old prefix
		  *  numbered i / amp_factor */
		for(j = 0; j < amp_factor; j ++) {
			new_prefix_arr[i + j] = prefix_arr[i / amp_factor];

			/**< Transform only the valid bytes */
			int bytes_to_transform = prefix_arr[i / amp_factor].depth / 8;

			for(k = 0; k < bytes_to_transform; k ++) {
				int old_byte = new_prefix_arr[i + j].bytes[k];
				int new_byte = perm_arr[j].P[old_byte];

				new_prefix_arr[i + j].bytes[k] = new_byte;
			}
		}
	}

	return new_prefix_arr;
}

inline uint32_t fastrand(uint64_t* seed)
{
    *seed = *seed * 1103515245 + 12345;
    return (uint32_t)(*seed >> 32);
}

/**< Generate probe IPv6 addresses from prefixes */
struct ipv6_addr *ipv6_gen_addrs(int num_addrs,
	struct ipv6_prefix *prefix_arr, int num_prefixes)
{
	assert(num_addrs > 0 && prefix_arr != NULL && num_prefixes > 0);

	struct ipv6_addr *addr_arr;
	int addr_mem_size = num_addrs * sizeof(struct ipv6_addr);

	addr_arr = hrd_malloc_socket(PROBE_ADDR_SHM_KEY, addr_mem_size, 0);

	/**< Generate addresses using randomly chosen prefixes */
	int i, j;

	for(i = 0; i < num_addrs; i ++) {
		int prefix_id = rand() % num_prefixes;

		for(j = 0; j < IPV6_ADDR_LEN; j ++) {
			addr_arr[i].bytes[j] = prefix_arr[prefix_id].bytes[j];
		}
	}

	return addr_arr;
}

void ipv6_print_prefix(struct ipv6_prefix *prefix)
{
	int i;
	printf("depth: %d, bytes: ", prefix->depth);
	for(i = 0; i < IPV6_ADDR_LEN; i ++) {
		printf("%d ", prefix->bytes[i]);
	}

	printf(" dst_port: %d\n", prefix->dst_port);
}

void ipv6_print_addr(struct ipv6_addr *addr)
{
	int i;
	for(i = 0; i < IPV6_ADDR_LEN; i ++) {
		printf("%d ", addr->bytes[i]);
	}

	printf("\n");
}

/**< Generate N different permutations of 0, ..., 255 */
struct ipv6_perm *ipv6_gen_perms(int N)
{
	struct ipv6_perm *res = malloc(N * sizeof(struct ipv6_perm));
	assert(res != 0);

	int i, j;
	for(i = 0; i < N; i ++) {
		/**< Generate the ith permutation */
		for(j = 0; j < 256; j ++) {
			res[i].P[j] = j;
		}

		/**< The 1st permutation returned is an identity permutation */
		if(i == 0) {
			continue;
		}

		for(j = 255; j >= 0; j --) {
			int k = rand() % (j + 1);
			uint8_t temp = res[i].P[j];
			res[i].P[j] = res[i].P[k];
			res[i].P[k] = temp;
		}
	}

	return res;
}
