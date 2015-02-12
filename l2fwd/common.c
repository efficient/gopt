/* Common functions used in DPDK code  */
#include "main.h"

void check_all_ports_link_status(uint8_t port_num, int portmask)
{
	uint8_t port_id;
	struct rte_eth_link link;

	printf("\nChecking link status\n");
	
	for (port_id = 0; port_id < port_num; port_id++) {
		if (!ISSET(portmask, port_id))
			continue;

		memset(&link, 0, sizeof(struct rte_eth_link));
		rte_eth_link_get(port_id, &link);
			
		if (link.link_status) {
			printf("Port %d Link Up \n", port_id);
		} else {
			printf("Port %d Link Down!\n", port_id);
		}
	}
}


struct rte_mempool *mempool_init(char *name, int socket_id)
{
	struct rte_mempool *ret = rte_mempool_create(name, 
		NB_MBUF, MBUF_SIZE, NB_MBUF_CACHE,
		sizeof(struct rte_pktmbuf_pool_private),
		rte_pktmbuf_pool_init, NULL, rte_pktmbuf_init, NULL,
		socket_id, 0);
	
	CPE(ret == NULL, "rte_mempool_create failed\n")
	return ret;
}

// FAC is a cycles-to-nanoseconds conversion factor
void micro_sleep(double us, double cycles_to_ns_fac)
{
	LL required_diff = (us * 1000) / cycles_to_ns_fac;
	LL start_tsc = rte_rdtsc();
	LL end_tsc = start_tsc;

	while(end_tsc - start_tsc <= required_diff) {
		end_tsc = rte_rdtsc();
	}
}

int count_active_lcores(void)
{
	int lid, ret = 0;
	for(lid = 0; lid < RTE_MAX_LCORE; lid++) {
		if(rte_lcore_is_enabled(lid)) {
			ret ++;
		}
	}
	return ret;
}

// Get a zero-based identifier for an lcore on a socket
// Rank of an lcore is the number of active lcores with IDs stricly
// smaller than its lcore_id, on specified socket_id
int get_lcore_rank(int lcore_id, int socket_id)
{
	int rank = -1, lid;
	for(lid = 0; lid <= lcore_id; lid ++) {
		if(rte_lcore_is_enabled(lid) && 
			rte_lcore_to_socket_id(lid) == (unsigned) socket_id) {
			rank ++;
		}
	}
	return rank;	
}

int get_lcore_ranked_n(int n, int socket_id)
{
	int lid;
	for(lid = 0; lid < RTE_MAX_LCORE; lid ++) {
		if(get_lcore_rank(lid, socket_id) == n) {
			return lid;
		}
	}

	fprintf(stderr, "No lcore of rank %d on socket %d exists. Exiting.\n",
		n, socket_id);
	exit(-1);
}


int count_active_lcores_on_socket(int socket_id)
{
	int active_lcores = 0;
	int i = 0;
	for(i = 0; i < RTE_MAX_LCORE; i++) {
		if(rte_lcore_is_enabled(i) && rte_lcore_to_socket_id(i) == (unsigned) socket_id) {
			active_lcores ++;
		}
	}
	return active_lcores;
}

// On CPUs that use an I/O Hub, rte_eth_dev_socket_id(port_id) returns -1.
// So, we assign a socket manually using macaddr. XIA-specific.
int get_socket_id_from_macaddr(int port_id)
{
	struct ether_addr macaddr;
	rte_eth_macaddr_get(port_id, &macaddr);

	int lsb = macaddr.addr_bytes[5];
	
	// Least byte of mac addr of xge0 and xge1 on xia-router1 and xia-router0
	if(lsb == 0x36 || lsb == 0x37 || lsb == 0x44 || lsb == 0x45) {
		return 0;
	}

	// Least byte of mac addr of xge2 and xge3 on xia-router1 and xia-router0
	if(lsb == 0xA8 || lsb == 0xA9 || lsb == 0x0A || lsb == 0x0B) {
		return 1;
	}

	red_printf("Unexpected mac addr ending: %x\n", lsb);
	exit(-1);
}

// Return the client lcore responsible for queue #queue_id on port #port_id
// XIA-specific
int client_port_queue_to_lcore(int port_id, int queue_id)
{
	assert(port_id <= 3 && port_id >= 0);
	assert(queue_id <= 2 && queue_id >=0);

	int mapping[4][3] = {{0, 2, 4},		// xge0: lcores 0, 2, 4
						 {6, 8, 10},		// xge1: lcores 6, 8, 10
						 {1, 3, 5},		// xge2: lcores 1, 3, 5
						 {7, 9, 11}};	// xge3: lcores 7, 9, 11

	return mapping[port_id][queue_id];
}

void print_ether_hdr(struct ether_hdr *eth_hdr)
{
	int i;
	printf("Ether hdr:\n");
	printf("\tDst mac: ");
	for(i = 0; i < 6; i ++) {
		printf("%x ", eth_hdr->d_addr.addr_bytes[i]);
	}
	printf("\n\tSrc mac: ");
	for(i = 0; i < 6; i ++) {
		printf("%x ", eth_hdr->s_addr.addr_bytes[i]);
	}
	printf("\n");
}

inline int
is_valid_ipv4_pkt(struct ipv4_hdr *pkt, uint32_t link_len)
{
	/* From http://www.rfc-editor.org/rfc/rfc1812.txt section 5.2.2 */
	/*   
	 * 1. The packet length reported by the Link Layer must be large
	 * enough to hold the minimum length legal IP datagram (20 bytes).
	 */
	if (link_len < sizeof(struct ipv4_hdr)) {
		red_printf("Invalid IPv4 packet: len < sizeof(ipv4_hdr)\n");
		return -1;
	}

	/* 2. The IP checksum must be correct. */
	/* this is checked in H/W */

	/*   
	 * 3. The IP version number must be 4. If the version number is not 4
	 * then the packet may be another version of IP, such as IPng or
	 * ST-II.
	 */
	if (((pkt->version_ihl) >> 4) != 4) {
		red_printf("Invalid IP version number\n");
		return -3;
	}
	/*   
	 * 4. The IP header length field must be large enough to hold the
	 * minimum length legal IP datagram (20 bytes = 5 words).
	 */
	if ((pkt->version_ihl & 0xf) < 5) {
		red_printf("Invalid IP header length field\n");
		return -4;
	}

	/*   
	 * 5. The IP total length field must be large enough to hold the IP
	 * datagram header, whose length is specified in the IP header length
	 * field.
	 */
	if (rte_cpu_to_be_16(pkt->total_length) < sizeof(struct ipv4_hdr)) {
		red_printf("Invalid IP total length field\n");
		return -5;
	}

	return 0;
}

float get_sleep_time(void)
{
	FILE *fp;
	fp = fopen("sleep_time", "r");
	if(fp == NULL) {
		red_printf("get_sleep_time failed to open sleep_time file.\n");
		return 0;
	}

	char sleep_buf[100] = {0};
	int num_read = fread(sleep_buf, 1, 10, fp);
	if(num_read == 0) {
		red_printf("get_sleep_time failed to read sleep_time file.\n");
		return 0;
	}
	
	fclose(fp);

	return atof(sleep_buf);
}
