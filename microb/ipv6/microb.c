#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/types.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include <rte_common.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_ip.h>
#include <rte_lpm.h>
#include <rte_lpm6.h>
#include <rte_string_fns.h>
#include <rte_ether.h>
#include <rte_ethdev.h>

#define NB_SOCKETS (2)
#define NB_MBUF (8192)
#define MEMPOOL_CACHE_SIZE 256
#define MBUF_SIZE (2048 + sizeof(struct rte_mbuf) + RTE_PKTMBUF_HEADROOM)

static int numa_on = 1;

static struct rte_mepool *pktmbuf_pool[NB_SOCKETS];

struct ipv6_fwd_route {
	uint8_t ip[16];
	uint8_t depth;
	uint8_t if_out;
};

static struct ipv6_fwd_route ipv6_fwd_route_array[] = {
	{{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}, 48, 0},
	{{2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}, 48, 1},
	{{3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}, 48, 2},
	{{4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}, 48, 3},
	{{5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}, 48, 4},
	{{6,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}, 48, 5},
	{{7,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}, 48, 6},
	{{8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}, 48, 7},
};

#define IPV6_FWD_NUM_ROUTES \
	(sizeof(ipv6_fwd_route_array) / sizeof(ipv6_fwd_route_array[0]))

#define IPV6_FWD_LPM_MAX_RULES (1024)
#define IPV6_FWD_LPM_NUMBER_TBL8S (1 << 16)

typedef struct rte_lpm6 lookup6_struct_t;
static lookup6_struct_t *ipv6_fwd_lookup_struct[NB_SOCKETS];

static void
setup_lpm(int socketid)
{
	struct rte_lpm6_config config;
	unsigned i;
	int ret;
	char s[64];

	/* create the LPM6 table */
	rte_snprintf(s, sizeof(s), "IPV6_FWD_LPM_%d", socketid);
	
	config.max_rules = IPV6_FWD_LPM_MAX_RULES;
	config.number_tbl8s = IPV6_FWD_LPM_NUMBER_TBL8S;
	config.flags = 0;
	ipv6_fwd_lookup_struct[socketid] = rte_lpm6_create(s, socketid,
				&config);
	if (ipv6_fwd_lookup_struct[socketid] == NULL)
		rte_exit(EXIT_FAILURE, "Unable to create the fwd LPM table"
				" on socket %d\n", socketid);

	/* populate the LPM table */
	for (i = 0; i < IPV6_FWD_NUM_ROUTES; i++) {
		ret = rte_lpm6_add(ipv6_fwd_lookup_struct[socketid],
			ipv6_fwd_route_array[i].ip,
			ipv6_fwd_route_array[i].depth,
			ipv6_fwd_route_array[i].if_out);

		if (ret < 0) {
			rte_exit(EXIT_FAILURE, "Unable to add entry %u to the "
				"fwd LPM table on socket %d\n",
				i, socketid);
		}

		printf("LPM: Adding route %s / %d (%d)\n",
			"IPV6",
			ipv6_fwd_route_array[i].depth,
			ipv6_fwd_route_array[i].if_out);
	}
}

static void
init(unsigned nb_mbuf)
{
	unsigned lcoreid;
	int socketid;
	char s[64];

	for (lcoreid = 0; lcoreid < RTE_MAX_LCORE; lcoreid++) {
		if (rte_lcore_is_enabled(lcoreid) == 0)
			continue;

		if (numa_on)
			socketid = rte_lcore_to_socket_id(lcoreid);
		else
			socketid = 0;
		if (pktmbuf_pool[socketid] == NULL) {
			rte_snprintf(s, sizeof(s), "mbuf_pool_%d", socketid);
			pktmbuf_pool[socketid] =
				rte_mempool_create(s, nb_mbuf, MBUF_SIZE, MEMPOOL_CACHE_SIZE,
								   sizeof(struct rte_pktmbuf_pool_private),
								   rte_pktmbuf_pool_init, NULL,
								   rte_pktmbuf_init, NULL,
								   socketid, 0);
			if (pktmbuf_pool[socketid] == NULL)
				rte_exit(EXIT_FAILURE,
						 "cannot init mbuf pool on socket %d\n", socketid);
			else
				printf("allocated mbuf pool on socket %d\n", socketid);
			setup_lpm(socketid);
		}
	}
}

int
main(int argc, char **argv)
{
	char *eal_argv[] = {
		argv[0],
		"-c", "1",
		"-n", "4",
	};
	int eal_argc = sizeof(eal_argv) / sizeof(eal_argv[0]);

	rte_set_log_level(RTE_LOG_NOTICE);
	if (rte_eal_init(eal_argc, eal_argv) < 0)
		rte_exit(EXIT_FAILURE, "invalid EAL parameters\n");
	
	init(NB_MBUF);
	printf("lookup struct initialized\n");

	return 0;
}
