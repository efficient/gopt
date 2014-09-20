#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>
#include <assert.h>

#include <rte_common.h>
#include <rte_cycles.h>
#include <rte_prefetch.h>
#include <rte_ip.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>

#include "fpp.h"
#include "util.h"
#include "cuckoo.h"

#define GOTO 0

// sizeof(rte_mbuf) = 64, RTE_PKTMBUF_HEADROOM = 128
#define MBUF_SIZE (2048 + sizeof(struct rte_mbuf) + RTE_PKTMBUF_HEADROOM)
#define DEFAULT_PACKET_LENGTH 60
#define NB_MBUF   8192
#define NB_MBUF_CACHE 512

#define RX_PTHRESH 8 /**< Default values of RX prefetch threshold reg. */
#define RX_HTHRESH 8 /**< Default values of RX host threshold reg. */
#define RX_WTHRESH 4 /**< Default values of RX write-back threshold reg. */
#define DEFAULT_NIC_RX_FREE_THRESH 64

#define TX_PTHRESH 36 /**< Default values of TX prefetch threshold reg. */
#define TX_HTHRESH 0  /**< Default values of TX host threshold reg. */
#define TX_WTHRESH 0  /**< Default values of TX write-back threshold reg. */

// Symbols for indexing into a latency measurement array
#define MSR_START 0
#define MSR_END 1
#define MSR_SAMPLES 2
#define MSR_TOT 3

// Configurable number of RX/TX ring descriptors
#define NUM_RX_DESC 512
#define NUM_TX_DESC 512

#define GHZ_CPS 1000000000
// Cycles to nanoseconds conversion constants
#define C_FAC ((double) GHZ_CPS / XIA_R0_CPS)
#define S_FAC ((double) GHZ_CPS / XIA_R2_CPS)

// All xia hardware specific features need to be prefixed by XIA_
#define XIA_R0_PORT_MASK 0x3	// xge0,1
#define XIA_R0_CPS 2270000000	// Client cycles per second

#define XIA_R2_PORT_MASK 0xf	// xge0,1,2,3
#define XIA_R2_CPS 2700000000	// Server cycles per second

// On all xia-router* machines, even numbered lcores are on socket 0
#define LCORE_TO_SOCKET(lcore) (lcore % 2)

// Application-specific RX/TX burst size for the server
#define MAX_SRV_BURST 16

/**
 * Per-lcore, per-port statistics:
 * The server process on each lcore creates a separate instance of
 * lcore_port_info for each port. The total number of packets transmitted
 * is collected in the nb_tx_all_ports field for port #0 (this does not
 * require that port #0 is enabled.
 */
struct lcore_port_info {
	struct rte_mbuf *mbufs[MAX_SRV_BURST];
	int nb_buf;
	int nb_tx;
	int nb_tx_fail;			// Port not found for a dst mac
	int nb_rx;

	int nb_tx_all_ports;
	int queue_id;
};

struct rte_mempool *mempool_init(char *name, int socket_id);

int client_port_queue_to_lcore(int port_id, int queue_id);
int count_active_lcores(void);
int get_lcore_rank(int lcore_id, int socket_id);
int get_lcore_ranked_n(int n, int socket_id);
int count_active_lcores_on_socket(int socket_id);
int get_socket_id_from_macaddr(int port_id);

void check_all_ports_link_status(uint8_t port_num, int portmask);

void run_server(struct cuckoo_bucket *ht_index);
void run_client(int client_id, unsigned long long *mac_addrs,
	 struct rte_mempool **l2fwd_pktmbuf_pool);

void micro_sleep(double us, double cycles_to_ns_fac);

void print_ether_hdr(struct ether_hdr *eth_hdr);

float get_sleep_time(void);

