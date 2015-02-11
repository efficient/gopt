#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "util.h"
#include "ipv4.h"


int ipv4_bitcount(int n);
int *ipv4_get_active_ports(int portmask);

/**< Count the number of 1-bits in n */
int ipv4_bitcount(int n)
{
	int count = 0;
	while(n > 0) {
		count ++;
		n = n & (n - 1);
	}
	return count;
}

/**< Returns an array containing the port numbers of all active ports */
int *ipv4_get_active_ports(int portmask)
{
	int num_active_ports = ipv4_bitcount(portmask);
	int *active_ports = (int *) malloc(num_active_ports * sizeof(int));
	int pos = 0, i;
	for(i = 0; i < IPv4_MAX_ETHPORTS; i++) {
		if(IPv4_ISSET(portmask, i)) {
			active_ports[pos] = i;
			pos ++;
		}
	}
	assert(pos == num_active_ports);
	return active_ports;
}

/**< Read IPv4 prefixes from a file */
struct ipv4_prefix *ipv4_read_prefixes(const char *prefixes_file,
	int *num_prefixes)
{
	assert(prefixes_file != NULL && num_prefixes != NULL);

	FILE *prefix_fp = fopen(prefixes_file, "r");
	assert(prefix_fp != NULL);

	fscanf(prefix_fp, "%d", num_prefixes);
	assert(*num_prefixes > 0);
	printf("ipv4: Reading %d prefixes\n", *num_prefixes);

	int prefix_mem_size = *num_prefixes * sizeof(struct ipv4_prefix);
	struct ipv4_prefix *prefix_arr = malloc(prefix_mem_size);
	assert(prefix_arr != NULL);

	int i, j;
	for(i = 0; i < *num_prefixes; i ++) {
		/**< A prefix is formatted as <depth> <bytes 0 ... 15> <dst port> */
		fscanf(prefix_fp, "%d", &prefix_arr[i].depth);

		for(j = 0; j < IPV4_ADDR_LEN; j ++) {
			int new_byte;
			fscanf(prefix_fp, "%d", &new_byte);
			assert(new_byte >= 0 && new_byte <= 255);

			prefix_arr[i].bytes[j] = new_byte;
		}

		fscanf(prefix_fp, "%d", &prefix_arr[i].dst_port);
	}

	return prefix_arr;	
}

/**< Generate IPv4 prefixes randomly */
struct ipv4_prefix *ipv4_gen_rand_prefixes(int num_prefixes)
{
	assert(num_prefixes > 0);

	int prefix_mem_size = num_prefixes * sizeof(struct ipv4_prefix);
	struct ipv4_prefix *prefix_arr = malloc(prefix_mem_size);
	assert(prefix_arr != NULL);

	int i, j;
	for(i = 0; i < num_prefixes; i ++) {

		/**< 97% of real-world IPv4 prefixes are <= 24 bits */
		prefix_arr[i].depth = 24;
		if(rand() % 100 <= 3) {
			prefix_arr[i].depth += rand() % 8;
		}

		for(j = 0; j < IPV4_ADDR_LEN; j ++) {
			prefix_arr[i].bytes[j] = rand() % 256;
		}

		prefix_arr[i].dst_port = rand() % 256;
	}

	return prefix_arr;
}

/**< Increase the number of prefixes in prefix_arr */
struct ipv4_prefix *ipv4_amp_prefixes(struct ipv4_prefix *prefix_arr,
	int num_prefixes, int amp_factor)
{
	int mem_size = num_prefixes * amp_factor * sizeof(struct ipv4_prefix);
	struct ipv4_prefix *new_prefix_arr = malloc(mem_size);
	assert(new_prefix_arr != NULL);

	struct ipv4_perm *perm_arr = ipv4_gen_perms(amp_factor);

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

/**< Generate probe IPv4 addresses from prefixes. This method chooses prefixes
  *  at random that have a valid match in the lpm. */
uint32_t *ipv4_gen_addrs(int lcore_id, struct rte_lpm *lpm)
{
	assert(lpm != NULL);
	int i;
	uint8_t next_hop;
	uint64_t seed = 0xdeadbeef;
	LL tries = 0;

	/**< Move fastrand so that clients don't get the same IP addresses */
	int fastrand_offset = (lcore_id / 2) * IPV4_NUM_ADDRS;
	printf("\tOffsetting fastrand by %d\n", fastrand_offset);
	for(i = 0; i < fastrand_offset; i ++) {
		fastrand(&seed);
	}

	uint32_t *addr_arr = malloc(IPV4_NUM_ADDRS * sizeof(uint32_t));

	for(i = 0; i < IPV4_NUM_ADDRS; i ++) {
		tries ++;
		uint32_t ip = (uint32_t) fastrand(&seed);

		if(rte_lpm_lookup(lpm, ip, &next_hop) == 0) {
			addr_arr[i] = ip;
		} else {
			i --;		/**< Try this value of i again */
		}
	}

	return addr_arr;
}

void ipv4_print_prefix(struct ipv4_prefix *prefix)
{
	int i;
	printf("depth: %d, bytes: ", prefix->depth);
	for(i = 0; i < IPV4_ADDR_LEN; i ++) {
		printf("%d ", prefix->bytes[i]);
	}

	printf(" dst_port: %d\n", prefix->dst_port);
}

void ipv4_print_addr(struct ipv4_addr *addr)
{
	int i;
	for(i = 0; i < IPV4_ADDR_LEN; i ++) {
		printf("%d ", addr->bytes[i]);
	}

	printf("\n");
}

/**< Generate N different permutations of 0, ..., 255 */
struct ipv4_perm *ipv4_gen_perms(int N)
{
	struct ipv4_perm *res = malloc(N * sizeof(struct ipv4_perm));
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

/**< Initialize an IPv4 lpm structure using prefixes from IPV4_PREFIX_FILE */
struct rte_lpm *ipv4_init(int portmask)
{
	int i, j;
	uint64_t seed = 0xdeadbeef;

	int num_active_ports = ipv4_bitcount(portmask);
	int *port_arr = ipv4_get_active_ports(portmask);

	struct rte_lpm *lpm = rte_lpm_create(0, IPV4_MAX_RULES);

	/**< Read the prefixes from a prefixes file */
	int num_prefixes;
	struct ipv4_prefix *prefix_arr = ipv4_read_prefixes(IPV4_PREFIX_FILE,
		&num_prefixes);

	for(i = 0; i < num_prefixes; i ++) {
		uint32_t prefix_ip = 0;

		for(j = 0; j < IPV4_ADDR_LEN; j ++) {
			prefix_ip += (prefix_arr[i].bytes[j] << (8 * (3 - j)));
		}

		int dst_port = port_arr[fastrand(&seed) % num_active_ports];

		int add_status = rte_lpm_add(lpm,
			prefix_ip, prefix_arr[i].depth, dst_port);
		if(add_status < 0) {
			printf("ipv4: Failed to add IPv4 prefix %d. Status = %d\n",
				i, add_status);
			exit(-1);
		}

		if(i % 20000 == 0) {
			printf("ipv4: Added %d IPs\n", i);
		}
	}
	
	return lpm;
}
