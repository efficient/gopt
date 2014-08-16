#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/types.h>
#include <sys/queue.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <setjmp.h>
#include <stdarg.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <unistd.h>
#include <assert.h>

#include <rte_common.h>
#include <rte_log.h>
#include <rte_memory.h>
#include <rte_memcpy.h>
#include <rte_memzone.h>
#include <rte_tailq.h>
#include <rte_eal.h>
#include <rte_per_lcore.h>
#include <rte_launch.h>
#include <rte_atomic.h>
#include <rte_cycles.h>
#include <rte_prefetch.h>
#include <rte_lcore.h>
#include <rte_per_lcore.h>
#include <rte_branch_prediction.h>
#include <rte_interrupts.h>
#include <rte_pci.h>
#include <rte_random.h>
#include <rte_debug.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_udp.h>
#include <rte_ethdev.h>
#include <rte_ring.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_byteorder.h>

#include "sizes.h"

#define LL long long

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

#define MAX_RX_QUEUE_PER_LCORE 16
#define MAX_TX_QUEUE_PER_PORT 16

#define ISSET(a, i) (a & (1 << i))
#define MAX(a, b) (a > b ? a : b)
#define htons(n) (((((unsigned short)(n) & 0xFF)) << 8) | (((unsigned short)(n) & 0xFF00) >> 8))



#define CPE2(val, msg, error, fault) \
	if(val) {fflush(stdout); rte_exit(EXIT_FAILURE, msg, error, fault);}
#define CPE(val, msg) \
	if(val) {fflush(stdout); rte_exit(EXIT_FAILURE, msg);}

#define GHZ_CPS 1000000000
// Cycles to nanoseconds conversion constants
#define C_FAC ((double) GHZ_CPS / XIA_R0_CPS)
#define S_FAC ((double) GHZ_CPS / XIA_R2_CPS)

#define LOG_CAP M_256
#define LOG_CAP_ M_256_
#define BASE_HT_LOG_SHM_KEY 1

// All xia hardware specific features need to be prefixed by XIA_
#define XIA_R0_PORT_MASK 0xc	// xge2,3
#define XIA_R0_CPS 2270000000	// Client cycles per second

#define XIA_R2_PORT_MASK 0xf0	// xge4,5,6,7
#define XIA_R2_CPS 2700000000	// Server cycles per second

// On all our machines, even numbered lcores are on socket 0
#define LCORE_TO_SOCKET(lcore) (lcore % 2)

struct rte_mempool *mempool_init(char *name, int socket_id);

int client_port_queue_to_lcore(int port_id, int queue_id);
int count_active_lcores(void);
int bitcount(int n);
int get_lcore_rank(int lcore_id, int socket_id);
int get_lcore_ranked_n(int n, int socket_id);
int *get_active_ports(int portmask);
int count_active_lcores_on_socket(int socket_id);
int get_socket_id_from_macaddr(int port_id);

void red_printf(const char *format, ...);
void print_mac(int port_id, struct ether_addr macaddr);
void check_all_ports_link_status(uint8_t port_num, int portmask);
void print_buf(char *A, int n);

void run_server(int *ht_log, struct rte_mempool **l2fwd_pktmbuf_pool);
void run_client(int client_id, int *ht_log, struct rte_mempool **l2fwd_pktmbuf_pool);
int *shm_alloc(int key, int cap);

inline uint32_t fastrand(uint64_t* seed);
void micro_sleep(int us, double cycles_to_ns_fac);

void set_mac(uint8_t *mac_ptr, LL mac_addr);
void swap_mac(uint8_t *src_mac_ptr, uint8_t *dst_mac_ptr);
void print_ether_hdr(struct ether_hdr *eth_hdr);

#define NUM_ACCESSES 0
