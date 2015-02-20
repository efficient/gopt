#include "main.h"

/**< Only immutable information should be global
  *  xia-router2 xge0,1,2,3,4,5,6,7 */
LL src_mac_arr[8] = {0x6c10bb211b00, 0x6d10bb211b00, 0x64d2bd211b00, 0x65d2bd211b00,
					 0xc8a610ca0568, 0xc9a610ca0568, 0xa2a610ca0568, 0xa3a610ca0568};

/**< xia-router0 xge0,1    xia-router1 xge0,1
  *  xia-router0 xge2,3    xia-router1 xge2,3 */
LL dst_mac_arr[8] = {0x36d3bd211b00, 0x37d3bd211b00, 0x44d7a3211b00, 0x45d7a3211b00,
					 0xa8d6a3211b00, 0xa9d6a3211b00, 0x0ad7a3211b00, 0x0bd7a3211b00};

/**
 * Enque a packet for transmission on a port. The per-port rx/tx/buffering 
 * statistics, and the queue to use for transmission are kept in the 
 * lcore_port_info structure.
 */ 
void send_packet(struct rte_mbuf *pkt, int port_id, 
	struct lcore_port_info *lp_info)
{
	int i;

	if(unlikely(!ISSET(XIA_R2_PORT_MASK, port_id))) {
		red_printf("TX on invalid port!. Exiting.\n");
		exit(-1);
	}

	int tot_buffered = lp_info[port_id].nb_buf;

	lp_info[port_id].mbufs[tot_buffered] = pkt;
	tot_buffered ++;

	/**< TX when a sufficient number of packets are buffered */
	if(unlikely(tot_buffered == MAX_SRV_BURST)) {
		int queue_id = lp_info[port_id].queue_id;
		int nb_tx_new = rte_eth_tx_burst(port_id, queue_id, 
			lp_info[port_id].mbufs, MAX_SRV_BURST);

		/**< Free unsent packets */
		for(i = nb_tx_new; i < MAX_SRV_BURST; i ++) {
			rte_pktmbuf_free(lp_info[port_id].mbufs[i]);
		}

		lp_info[port_id].nb_tx += nb_tx_new;
		lp_info[0].nb_tx_all_ports += nb_tx_new;
		
		lp_info[port_id].nb_buf = 0;
		
	} else {
		lp_info[port_id].nb_buf = tot_buffered;
	}
}

void process_batch_nogoto(struct rte_mbuf **pkts, int nb_pkts,
	struct lcore_port_info *lp_info,
	const struct mac_ints *mac_ints_arr)
{
	int batch_index = 0;

	/**< sizeof(ether_hdr) + sizeof(ipv4_hdr) is 34 --> 36 for 4 byte align */
	int hdr_size = 36;

	foreach(batch_index, nb_pkts) {
		struct ether_hdr *eth_hdr;

		if(batch_index != nb_pkts - 1) {
			rte_prefetch0(pkts[batch_index + 1]->pkt.data);
		}

		eth_hdr = (struct ether_hdr *) pkts[batch_index]->pkt.data;

		/**< Extract the client's request and compute response */
		int *req = (int *) (rte_pktmbuf_mtod(pkts[batch_index], char *) +
			hdr_size + 20);

		int dst_port = req[0] & 3;

		/**< TX boilerplate: use the computed next_hop for L2 src and dst. */
		int *mac_ints_dst = (int *) eth_hdr;
		mac_ints_dst[0] = mac_ints_arr[dst_port].chunk[0];
		mac_ints_dst[1] = mac_ints_arr[dst_port].chunk[1];
		mac_ints_dst[2] = mac_ints_arr[dst_port].chunk[2];

		/**< Garble dst MAC to reduce RX load on clients */
		eth_hdr->d_addr.addr_bytes[0] += ((req[0] >> 8) & 0xff);

		/**< Send out the packet based on the value of resp */
		send_packet(pkts[batch_index], dst_port, lp_info);
	}
}

void run_server(void)
{
	int i;

	int lcore_id = rte_lcore_id();
	int socket_id = rte_lcore_to_socket_id(lcore_id);
	assert(socket_id == 0);

	int queue_id = get_lcore_rank(lcore_id, socket_id);
	printf("Server on lcore %d. Queue Id = %d\n", lcore_id, queue_id);

	/**< This check is required because the process_batch_gpu function
	  *  uses the GPU's resp to determine the port to send the packet
	  *  on as follows: port = resp % 4 */
	assert(XIA_R2_PORT_MASK == 0xf);

	int num_active_ports = bitcount(XIA_R2_PORT_MASK);
	int *port_arr = get_active_bits(XIA_R2_PORT_MASK);

	/**< Construct the mac ints for all 4 ports. This allows us to set the
	  *  Ethernet header during TX in 3 integer copies. */
	assert(num_active_ports <= 4);
	struct mac_ints mac_ints_arr[4];
	for(i = 0; i < 4; i ++) {
		uint8_t *hack_bytes = (uint8_t *) mac_ints_arr[i].chunk;
		set_mac(&hack_bytes[0], dst_mac_arr[i]);
		set_mac(&hack_bytes[6], src_mac_arr[i]);
	}
	
	/**< Initialize the per-port info for this lcore */
	struct lcore_port_info lp_info[RTE_MAX_ETHPORTS];
	memset(lp_info, 0, RTE_MAX_ETHPORTS * sizeof(struct lcore_port_info));
	for(i = 0; i < RTE_MAX_ETHPORTS; i ++) {
		lp_info[i].queue_id = queue_id;
	}

	struct rte_mbuf *rx_pkts_burst[MAX_SRV_BURST];
	int port_index = 0;

	// Init measurement variables
	LL tput_tsc[2], brst_sz_msr[4];
	tput_tsc[0] = rte_rdtsc();
	memset(brst_sz_msr, 0, 4 * sizeof(LL));

	while (1) {
		int port_id = port_arr[port_index];	// The port to use in this iteration
		int nb_rx_new = 0, tries = 0;
		
		/**< Lcores *cannot* wait for a fixed number of packets from a port.
		  *  If we do this, the port mysteriously runs out of RX desc */
		while(nb_rx_new < MAX_SRV_BURST && tries < 5) {
			nb_rx_new += rte_eth_rx_burst(port_id, queue_id, 
				&rx_pkts_burst[nb_rx_new], MAX_SRV_BURST - nb_rx_new);
			tries ++;
		}
		
		if(nb_rx_new == 0) {
			port_index = (port_index + 1) < num_active_ports ? port_index + 1 : 0;
			continue;
		}

		/**< Measurements for burst size averaging */
		brst_sz_msr[MSR_SAMPLES] ++;
		brst_sz_msr[MSR_TOT] += nb_rx_new;
	
		lp_info[port_id].nb_rx += nb_rx_new;

		process_batch_nogoto(rx_pkts_burst, nb_rx_new, lp_info, mac_ints_arr);
		
		/**< STAT PRINTING */
		if (unlikely(lp_info[0].nb_tx_all_ports >= 10000000)) {
			tput_tsc[1] = rte_rdtsc();
			double nanoseconds = S_FAC * (tput_tsc[1] - tput_tsc[0]);
			double seconds = nanoseconds / GHZ_CPS;
			tput_tsc[0] = tput_tsc[1];

			red_printf("Lcore %d, total: %f\n", lcore_id, 
				lp_info[0].nb_tx_all_ports / seconds);

			/**< Reset all-port stats in case port 0 is disabled */
			lp_info[0].nb_tx_all_ports = 0;
			for(i = 0; i < RTE_MAX_ETHPORTS; i++) {
				if(ISSET(XIA_R2_PORT_MASK, i)) {
					printf("\tLcore: %d, port: %d: %f\n", lcore_id, i, 
						lp_info[i].nb_tx / seconds);
				}

				/**< Do not reset the nb_buf counter */
				lp_info[i].nb_tx = 0;
				lp_info[i].nb_rx = 0;
			}

			printf("\tLcore %d, Average TX burst size: %lld\n", lcore_id, 
				brst_sz_msr[MSR_TOT] / brst_sz_msr[MSR_SAMPLES]);
			printf("\n");

			memset(brst_sz_msr, 0, 4 * sizeof(LL));
			
		}

		port_index = (port_index + 1) % num_active_ports;
	}
}
