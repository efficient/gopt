#include "main.h"

#define MAX_SRV_BURST 16
// Only immutable information should be global
// xia-router2 xge0,1,2,3,4,5,6,7
LL src_mac_arr[8] = {0x6c10bb211b00, 0x6d10bb211b00, 0x64d2bd211b00, 0x65d2bd211b00,
					 0xc8a610ca0568, 0xc9a610ca0568, 0xa2a610ca0568, 0xa3a610ca0568};

// xia-router0 xge0,1    xia-router1 xge0,1    xia-router0 xge2,3    xia-router1 xge2,3
LL dst_mac_arr[8] = {0x36d3bd211b00, 0x37d3bd211b00, 0x44d7a3211b00, 0x45d7a3211b00,
					 0xa8d6a3211b00, 0xa9d6a3211b00, 0x0ad7a3211b00, 0x0bd7a3211b00};
	
void run_server(void)
{
	int i;
	LL nb_tx[RTE_MAX_ETHPORTS] = {0}, nb_rx[RTE_MAX_ETHPORTS] = {0}, nb_tx_all_ports = 0;

	int num_active_ports = bitcount(XIA_R2_PORT_MASK);
	int *port_arr = get_active_bits(XIA_R2_PORT_MASK);
	int port_index = 0;

	struct rte_mbuf *rx_pkts_burst[MAX_SRV_BURST];

	int lcore_id = rte_lcore_id();						// Lcore on which this server process runs
	int socket_id = rte_lcore_to_socket_id(lcore_id);

	int queue_id = get_lcore_rank(lcore_id, socket_id);

	printf("Server on lcore %d. Queue Id = %d\n", lcore_id, queue_id);

	struct ether_hdr *eth_hdr;
	struct ipv4_hdr *ip_hdr;
	void *src_mac_ptr, *dst_mac_ptr;

	// sizeof(ether_hdr) + sizeof(ipv4_hdr) is 34 --> 36 for 4 byte alignment
	int hdr_size = 36;
	uint64_t rss_seed = 0xdeadbeef;

	// Init measurement variables
	LL tput_tsc[2], brst_sz_msr[4];
	tput_tsc[0] = rte_rdtsc();
	memset(brst_sz_msr, 0, 4 * sizeof(LL));

	while (1) {
		int port_id = port_arr[port_index];	// The port to use in this iteration
		int nb_rx_new = 0, tries = 0;
		
		// Lcores *cannot* wait for a particular number of packets from a port.
		//  If we do this, the port mysteriously runs out of RX descriptors.
		while(nb_rx_new < MAX_SRV_BURST && tries < 5) {
			nb_rx_new += rte_eth_rx_burst(port_id, queue_id, 
				&rx_pkts_burst[nb_rx_new], MAX_SRV_BURST - nb_rx_new);
			tries ++;
		}
		
		if(nb_rx_new == 0) {
			port_index = (port_index + 1) < num_active_ports ? port_index + 1 : 0;
			continue;
		}
	
		nb_rx[port_id] += nb_rx_new;
		
		for(i = 0; i < nb_rx_new; i++) {
			// Boilerplate for TX pkt
			if(i != nb_rx_new - 1) {
				rte_prefetch0(rte_pktmbuf_mtod(rx_pkts_burst[i + 1], void *));
			}

			eth_hdr = rte_pktmbuf_mtod(rx_pkts_burst[i], struct ether_hdr *);
    		ip_hdr = (struct ipv4_hdr *) ((char *) eth_hdr + sizeof(struct ether_hdr));
			
			src_mac_ptr = &eth_hdr->s_addr.addr_bytes[0];
			dst_mac_ptr = &eth_hdr->d_addr.addr_bytes[0];
			swap_mac(src_mac_ptr, dst_mac_ptr);

			eth_hdr->ether_type = htons(ETHER_TYPE_IPv4);
	
			// These 3 fields of ip_hdr are required for RSS
    		ip_hdr->src_addr = fastrand(&rss_seed);
    		ip_hdr->dst_addr = fastrand(&rss_seed);
			ip_hdr->version_ihl = 0x40 | 0x05;

			rx_pkts_burst[i]->pkt.nb_segs = 1;
			rx_pkts_burst[i]->pkt.pkt_len = 60;
			rx_pkts_burst[i]->pkt.data_len = 60;

			// Actual code for data access
			int *req = (int *) (rte_pktmbuf_mtod(rx_pkts_burst[i], char *) + hdr_size);
			req[2] = req[1];
		}
	
		// Measurements for burst size averaging
		brst_sz_msr[MSR_SAMPLES] ++;
		brst_sz_msr[MSR_TOT] += nb_rx_new;
	
		int nb_tx_new = rte_eth_tx_burst(port_id, queue_id, rx_pkts_burst, nb_rx_new);

		// Free unsent packets
		for(i = nb_tx_new; i < nb_rx_new; i ++) {
			rte_pktmbuf_free(rx_pkts_burst[i]);
		}
		
		nb_tx[port_id] += nb_tx_new;
		nb_tx_all_ports += nb_tx_new;

		// STAT PRINTING
		if (unlikely(nb_tx_all_ports >= 10000000)) {
			tput_tsc[1] = rte_rdtsc();
			double nanoseconds = S_FAC * (tput_tsc[1] - tput_tsc[0]);
			tput_tsc[0] = tput_tsc[1];

			red_printf("Lcore %d, total: %f\n", lcore_id, 
				nb_tx_all_ports / (nanoseconds / GHZ_CPS));

			for(i = 0; i < RTE_MAX_ETHPORTS; i++) {
				if(ISSET(XIA_R2_PORT_MASK, i)) {
					printf("\tLcore: %d, port: %d: %f\n", lcore_id, i, nb_tx[i] / (nanoseconds / GHZ_CPS));
				}
			}
			

			printf("\tLcore %d, Average TX burst size: %lld\n", lcore_id, 
				brst_sz_msr[MSR_TOT] / brst_sz_msr[MSR_SAMPLES]);
			printf("\n");

			memset(brst_sz_msr, 0, 4 * sizeof(LL));

			memset(nb_rx, 0, RTE_MAX_ETHPORTS * sizeof(LL));
			memset(nb_tx, 0, RTE_MAX_ETHPORTS * sizeof(LL));
			nb_tx_all_ports = 0;
			
		}

		port_index = (port_index + 1) % num_active_ports;
	}
}
