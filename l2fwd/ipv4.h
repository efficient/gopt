#include "rte_lpm.h"

#define IPV4_ADDR_LEN 4
#define PROBE_ADDR_SHM_KEY 2
#define IPV4_MAX_RULES (2 * 1024 * 1024)

#define IPV4_PREFIX_FILE "../data_dump/ipv4/ipv4_java_out"

/**< Don't want to include DPDK headers */
#define IPv4_MAX_ETHPORTS 16
#define IPv4_ISSET(a, i) (a & (1 << i))

/**< Don't want to include main.h for XIA_R2_PORT_MASK */
#define IPv4_XIA_R2_PORT_MASK 0xf

struct ipv4_prefix {
	int depth;	/**< Number of bits required to match exactly */
	uint8_t bytes[IPV4_ADDR_LEN];
	int dst_port;
};

struct ipv4_addr {
	uint8_t bytes[IPV4_ADDR_LEN];
};

struct ipv4_perm {
	uint8_t P[256];
};

/**< Initialize the LPM tables for IPv4 */
struct rte_lpm *ipv4_init(int portmask);

/**< Read IPv4 prefixes from a file */
struct ipv4_prefix *ipv4_read_prefixes(const char *prefixes_file,
	int *num_prefixes);

/**< Generate IPv4 prefixes randomly */
struct ipv4_prefix *ipv4_gen_rand_prefixes(int num_prefixes);

/**< Increase the number of prefixes in prefix_arr. Returns a new
  *  array with num_prefixes * amp_factor prefixes */
struct ipv4_prefix *ipv4_amp_prefixes(struct ipv4_prefix *prefix_arr,
	int num_prefixes, int amp_factor);

/**< Generate probe IPv4 addresses from prefixes */
struct ipv4_addr *ipv4_gen_addrs(int num_addrs,
	struct ipv4_prefix *prefix_arr, int num_prefixes);

/**< Generate N different permutations of 0, ..., 255 */
struct ipv4_perm *ipv4_gen_perms(int N);

void ipv4_print_prefix(struct ipv4_prefix *prefix);
void ipv4_print_addr(struct ipv4_addr *addr);
