#include "main.h"

// Only immutable information should be global
// xia-router2 xge0,1,2,3,4,5,6,7
LL src_mac_arr[8] = {0x6c10bb211b00, 0x6d10bb211b00, 0x64d2bd211b00, 0x65d2bd211b00,
					 0xc8a610ca0568, 0xc9a610ca0568, 0xa2a610ca0568, 0xa3a610ca0568};

// xia-router0 xge0,1    xia-router1 xge0,1    xia-router0 xge2,3    xia-router1 xge2,3
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

	// TX when a sufficient number of packets are buffered
	if(unlikely(tot_buffered == MAX_SRV_BURST)) {
		int queue_id = lp_info[port_id].queue_id;
		int nb_tx_new = rte_eth_tx_burst(port_id, queue_id, 
			lp_info[port_id].mbufs, MAX_SRV_BURST);

		// Free unsent packets
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
	struct cuckoo_bucket *ht_index,
	struct lcore_port_info *lp_info, int port_id)
{
	int batch_index = 0;

	foreach(batch_index, nb_pkts) {
		int i;
		struct ether_hdr *eth_hdr;

		void *dst_mac_ptr;
		ULL dst_mac;
		int bkt_1, bkt_2, fwd_port = -1;

		if(batch_index != nb_pkts - 1) {
			rte_prefetch0(pkts[batch_index + 1]->pkt.data);
		}

		eth_hdr = (struct ether_hdr *) pkts[batch_index]->pkt.data;

		dst_mac_ptr = &eth_hdr->d_addr.addr_bytes[0];
		/**< We need the dst_mac for comparison with the key in hash-table */
		dst_mac = get_mac(eth_hdr->d_addr.addr_bytes);

		/**< Compute the 1st bucket using the full mac address */
		bkt_1 = CityHash32(dst_mac_ptr, 6) & NUM_BKT_;
		FPP_EXPENSIVE(&ht_index[bkt_1]);

		for(i = 0; i < 8; i ++) {
			if(SLOT_TO_MAC(ht_index[bkt_1].slot[i]) == dst_mac) {
				fwd_port = SLOT_TO_PORT(ht_index[bkt_1].slot[i]);
				break;
			}
		}

		/**< 2nd bucket is computed using the 1st bucket */
		if(fwd_port == -1) {
			bkt_2 = CityHash32((char *) &bkt_1, 4) & NUM_BKT_;
			FPP_EXPENSIVE(&ht_index[bkt_2]);

			for(i = 0; i < 8; i ++) {
				if(SLOT_TO_MAC(ht_index[bkt_2].slot[i]) == dst_mac) {
					fwd_port = SLOT_TO_PORT(ht_index[bkt_2].slot[i]);
					break;
				}
			}
		} 
			
		/**< Count failed packets and transmit */
		if(fwd_port == -1) {
			lp_info[port_id].nb_tx_fail ++;
			rte_pktmbuf_free(pkts[batch_index]);
		} else {
			set_mac(eth_hdr->d_addr.addr_bytes, dst_mac_arr[fwd_port]);
		
			/**< Reduce RX load on client: If the client sent a bad 
			  *  rc address, garble dst address */
			if(eth_hdr->s_addr.addr_bytes[0] == 0xef) {
				eth_hdr->d_addr.addr_bytes[0] ++;
			}
			set_mac(eth_hdr->s_addr.addr_bytes, src_mac_arr[port_id]);
			send_packet(pkts[batch_index], fwd_port, lp_info);
		}
	}
}

void run_server(struct cuckoo_bucket *ht_index)
{
	int i;

	int lcore_id = rte_lcore_id();
	int socket_id = rte_lcore_to_socket_id(lcore_id);
	assert(socket_id == 0);

	int queue_id = get_lcore_rank(lcore_id, socket_id);
	printf("Server on lcore %d. Queue Id = %d\n", lcore_id, queue_id);

	int num_active_ports = bitcount(XIA_R2_PORT_MASK);
	int *port_arr = get_active_bits(XIA_R2_PORT_MASK);
	
	/**< Initialize the per-port info for this lcore */
	struct lcore_port_info lp_info[RTE_MAX_ETHPORTS];
	memset(lp_info, 0, RTE_MAX_ETHPORTS * sizeof(struct lcore_port_info));
	for(i = 0; i < RTE_MAX_ETHPORTS; i ++) {
		lp_info[i].queue_id = queue_id;
	}

	struct rte_mbuf *rx_pkts_burst[MAX_SRV_BURST];
	int port_index = 0;

	/**< Init measurement variables */
	LL tput_tsc[2], brst_sz_msr[4];
	tput_tsc[0] = rte_rdtsc();
	memset(brst_sz_msr, 0, 4 * sizeof(LL));

	while (1) {
		int port_id = port_arr[port_index];	/**< Port for this iteration */
		int nb_rx_new = 0, tries = 0;
		
		/**< Lcores *cannot* wait for a fixed number of packets from a port.
		  * If we do this, the port mysteriously runs out of RX desc */
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

#if GOTO == 1
		process_batch_goto(rx_pkts_burst, 
			nb_rx_new, ht_index, lp_info, port_id);
#else
		process_batch_nogoto(rx_pkts_burst,
			nb_rx_new, ht_index, lp_info, port_id);
#endif
		
		/**< STAT PRINTING */
		if (unlikely(lp_info[0].nb_tx_all_ports >= 10000000)) {
			tput_tsc[1] = rte_rdtsc();
			double nanoseconds = S_FAC * (tput_tsc[1] - tput_tsc[0]);
			double seconds = nanoseconds / GHZ_CPS;
			tput_tsc[0] = tput_tsc[1];

			red_printf("Lcore %d, total: %f\n", lcore_id, 
				lp_info[0].nb_tx_all_ports / seconds);

			for(i = 0; i < RTE_MAX_ETHPORTS; i++) {
				if(ISSET(XIA_R2_PORT_MASK, i)) {
					printf("\tLcore: %d, port %d: S: %f, F: %f\n", lcore_id, i,
						lp_info[i].nb_tx / seconds, 
						lp_info[i].nb_tx_fail / seconds);
				}

				// Do not reset the nb_buf counter
				lp_info[i].nb_tx = 0;
				lp_info[i].nb_tx_fail = 0;

				lp_info[i].nb_rx = 0;
	
				lp_info[i].nb_tx_all_ports = 0;
			}

			printf("\tLcore %d, Average TX burst size: %lld\n", lcore_id, 
				brst_sz_msr[MSR_TOT] / brst_sz_msr[MSR_SAMPLES]);
			printf("\n");

			memset(brst_sz_msr, 0, 4 * sizeof(LL));
			
		}

		port_index = (port_index + 1) % num_active_ports;
	}
}
