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

void process_batch_goto(struct rte_mbuf **pkts, int nb_pkts,
                          uint64_t *rss_seed, struct cuckoo_bucket *ht_index,
                          struct lcore_port_info *lp_info, int port_id)
{
	int i[BATCH_SIZE];
	struct ether_hdr *eth_hdr[BATCH_SIZE];
	struct ipv4_hdr *ip_hdr[BATCH_SIZE];
	void *dst_mac_ptr[BATCH_SIZE];
	ULL dst_mac[BATCH_SIZE];
	int fwd_port[BATCH_SIZE];
	int bkt_2[BATCH_SIZE];
	int bkt_1[BATCH_SIZE];

	int I = 0;			// batch index
	void *batch_rips[BATCH_SIZE];		// goto targets
	int iMask = 0;		// No packet is done yet

	int temp_index;
	for(temp_index = 0; temp_index < BATCH_SIZE; temp_index ++) {
		batch_rips[temp_index] = &&fpp_start;
	}

fpp_start:

	fwd_port[I] = -1;

	if(I != nb_pkts - 1) {
		rte_prefetch0(pkts[I + 1]->pkt.data);
	}

	eth_hdr[I] = (struct ether_hdr *) pkts[I]->pkt.data;
	ip_hdr[I] = (struct ipv4_hdr *) ((char *) eth_hdr[I] + sizeof(struct ether_hdr));

	dst_mac_ptr[I] = &eth_hdr[I]->d_addr.addr_bytes[0];
	dst_mac[I] = get_mac(eth_hdr[I]->d_addr.addr_bytes);

	eth_hdr[I]->ether_type = htons(ETHER_TYPE_IPv4);

	// These 3 fields of ip_hdr are required for RSS
	ip_hdr[I]->src_addr = fastrand(rss_seed);
	ip_hdr[I]->dst_addr = fastrand(rss_seed);
	ip_hdr[I]->version_ihl = 0x40 | 0x05;

	pkts[I]->pkt.nb_segs = 1;
	pkts[I]->pkt.pkt_len = 60;
	pkts[I]->pkt.data_len = 60;

	bkt_1[I] = CityHash32(dst_mac_ptr[I], 6) & NUM_BKT_;
	FPP_PSS(&ht_index[bkt_1[I]], fpp_label_1, nb_pkts);
fpp_label_1:

	for(i[I] = 0; i[I] < 8; i[I] ++) {
		if(SLOT_TO_MAC(ht_index[bkt_1[I]].slot[i[I]]) == dst_mac[I]) {
			fwd_port[I] = SLOT_TO_PORT(ht_index[bkt_1[I]].slot[i[I]]);
			break;
		}
	}

	if(fwd_port[I] == -1) {
		bkt_2[I] = CityHash32((char *) &bkt_1[I], 4) & NUM_BKT_;
		FPP_PSS(&ht_index[bkt_2[I]], fpp_label_2, nb_pkts);
fpp_label_2:

		for(i[I] = 0; i[I] < 8; i[I] ++) {
			if(SLOT_TO_MAC(ht_index[bkt_2[I]].slot[i[I]]) == dst_mac[I]) {
				fwd_port[I] = SLOT_TO_PORT(ht_index[bkt_2[I]].slot[i[I]]);
				break;
			}
		}
	}

	if(fwd_port[I] == -1) {
		lp_info[port_id].nb_tx_fail ++;
		rte_pktmbuf_free(pkts[I]);
	} else {
		set_mac(eth_hdr[I]->s_addr.addr_bytes, src_mac_arr[port_id]);
		set_mac(eth_hdr[I]->d_addr.addr_bytes, dst_mac_arr[fwd_port[I]]);
		send_packet(pkts[I], fwd_port[I], lp_info);
	}

fpp_end:
	batch_rips[I] = &&fpp_end;
	iMask = FPP_SET(iMask, I); 
	if(iMask == (1 << nb_pkts) - 1) {
		return;
	}
	I = (I + 1) < nb_pkts ? I + 1 : 0;
	goto *batch_rips[I];
}

void process_batch_nogoto(struct rte_mbuf **pkts, int nb_pkts, 
	uint64_t *rss_seed, struct cuckoo_bucket *ht_index,
	struct lcore_port_info *lp_info, int port_id)
{
	int batch_index = 0;

	foreach(batch_index, nb_pkts) {
		int i;
		struct ether_hdr *eth_hdr;
		struct ipv4_hdr *ip_hdr;

		void *dst_mac_ptr;
		ULL dst_mac;
		int bkt_1, bkt_2, fwd_port = -1;

		if(batch_index != nb_pkts - 1) {
			rte_prefetch0(pkts[batch_index + 1]->pkt.data);
		}

		eth_hdr = (struct ether_hdr *) pkts[batch_index]->pkt.data;
		ip_hdr = (struct ipv4_hdr *) ((char *) eth_hdr + sizeof(struct ether_hdr));

		dst_mac_ptr = &eth_hdr->d_addr.addr_bytes[0];
		/** < We need the dst_mac for comparison with the key in hash-table */
		dst_mac = get_mac(eth_hdr->d_addr.addr_bytes);

		eth_hdr->ether_type = htons(ETHER_TYPE_IPv4);

		/** < RSS fields */
		ip_hdr->src_addr = fastrand(rss_seed);
		ip_hdr->dst_addr = fastrand(rss_seed);
		ip_hdr->version_ihl = 0x40 | 0x05;

		pkts[batch_index]->pkt.nb_segs = 1;
		pkts[batch_index]->pkt.pkt_len = 60;
		pkts[batch_index]->pkt.data_len = 60;

		/** < Compute the 1st bucket using the full mac address */
		bkt_1 = CityHash32(dst_mac_ptr, 6) & NUM_BKT_;
		FPP_EXPENSIVE(&ht_index[bkt_1]);

		for(i = 0; i < 8; i ++) {
			if(SLOT_TO_MAC(ht_index[bkt_1].slot[i]) == dst_mac) {
				fwd_port = SLOT_TO_PORT(ht_index[bkt_1].slot[i]);
				break;
			}
		}

		/** < 2nd bucket is computed using the 1st bucket */
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
			
		/** < Count failed packets and transmit */
		if(fwd_port == -1) {
			lp_info[port_id].nb_tx_fail ++;
			rte_pktmbuf_free(pkts[batch_index]);
		} else {
			set_mac(eth_hdr->s_addr.addr_bytes, src_mac_arr[port_id]);
			set_mac(eth_hdr->d_addr.addr_bytes, dst_mac_arr[fwd_port]);
			send_packet(pkts[batch_index], fwd_port, lp_info);
		}
	}
}

void run_server(struct cuckoo_bucket *ht_index)
{
	int i;
	uint64_t rss_seed = 0xdeadbeef;

	int lcore_id = rte_lcore_id();
	int socket_id = rte_lcore_to_socket_id(lcore_id);
	assert(socket_id == 0);

	int queue_id = get_lcore_rank(lcore_id, socket_id);
	printf("Server on lcore %d. Queue Id = %d\n", lcore_id, queue_id);

	int num_active_ports = bitcount(XIA_R2_PORT_MASK);
	int *port_arr = get_active_bits(XIA_R2_PORT_MASK);
	
	// Initialize the per-port info for this lcore
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

		// Measurements for burst size averaging
		brst_sz_msr[MSR_SAMPLES] ++;
		brst_sz_msr[MSR_TOT] += nb_rx_new;
	
		lp_info[port_id].nb_rx += nb_rx_new;

#if GOTO == 1
		process_batch_goto(rx_pkts_burst, 
			nb_rx_new, &rss_seed, ht_index, lp_info, port_id);
#else
		process_batch_nogoto(rx_pkts_burst,
			nb_rx_new, &rss_seed, ht_index, lp_info, port_id);
#endif
		
		// STAT PRINTING
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
