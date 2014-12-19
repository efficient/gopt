#define IPV6_ADDR_LEN 16
#define PROBE_ADDR_SHM_KEY 2

struct ipv6_prefix {
	int depth;	/**< Number of bits required to match exactly */
	uint8_t bytes[IPV6_ADDR_LEN];
	int dst_port;
};

struct ipv6_addr {
	uint8_t bytes[IPV6_ADDR_LEN];
};

/**< Read IPv6 prefixes from a file */
struct ipv6_prefix *ipv6_read_prefixes(const char *prefixes_file,
	int *num_prefixes);

/**< Generate probe IPv6 addresses from prefixes */
struct ipv6_addr *ipv6_gen_addrs(int num_addrs,
	struct ipv6_prefix *prefix_arr, int num_prefixes);

