#include "rte_lpm6.h"
#include "emmintrin.h"

#define IPV6_ADDR_LEN 16
#define PROBE_ADDR_SHM_KEY 2

#define IPV6_NUM_IPS (64 * 1024 * 1024)
#define IPV6_NUM_RAND_PREFIXES 200000
#define IPV6_RAND_PREFIXES_SEED 3185

/**< Number of tbl8s to request from DPDK lpm6 */
#define IPV6_NUM_TBL8 (1024 * 1024)

/**< Don't want to include main.h for XIA_R2_PORT_MASK */
#define IPV6_XIA_R2_PORT_MASK 0xf

struct ipv6_prefix {
	int depth;	/**< Number of bits required to match exactly */
	uint8_t bytes[IPV6_ADDR_LEN];
	int dst_port;
};

struct ipv6_addr {
	uint8_t bytes[IPV6_ADDR_LEN];
};

struct ipv6_perm {
	uint8_t P[256];
};

struct rte_lpm6 *ipv6_init(int portmask,
	struct ipv6_prefix **prefix_arr, int add_prefixes);

/**< Read IPv6 prefixes from a file */
struct ipv6_prefix *ipv6_read_prefixes(const char *prefixes_file,
	int *num_prefixes);

/**< Generate IPv6 prefixes randomly */
struct ipv6_prefix *ipv6_gen_rand_prefixes(int num_prefixes, int portmask);

/**< Increase the number of prefixes in prefix_arr. Returns a new
  *  array with num_prefixes * amp_factor prefixes */
struct ipv6_prefix *ipv6_amp_prefixes(struct ipv6_prefix *prefix_arr,
	int num_prefixes, int amp_factor);

/**< Generate probe IPv6 addresses from prefixes */
struct ipv6_addr *ipv6_gen_addrs(int num_addrs,
	struct ipv6_prefix *prefix_arr, int num_prefixes);

/**< Generate N different permutations of 0, ..., 255 */
struct ipv6_perm *ipv6_gen_perms(int N);

void ipv6_print_prefix(struct ipv6_prefix *prefix);
void ipv6_print_addr(struct ipv6_addr *addr);

/**XXX: Copied from DPDK rte_memcpy.h
 * Copy 16 bytes from one location to another using optimised SSE
 * instructions. The locations should not overlap.
 *
 * @param dst
 *   Pointer to the destination of the data.
 * @param src
 *   Pointer to the source data.
 */
static inline void
ipv6_mov16(uint8_t *dst, const uint8_t *src)
{
    __m128i reg_a;
    asm volatile (
        "movdqu (%[src]), %[reg_a]\n\t"
        "movdqu %[reg_a], (%[dst])\n\t"
        : [reg_a] "=x" (reg_a)
        : [src] "r" (src),
          [dst] "r"(dst)
        : "memory"
    );
}

