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

/**< Takes a pointer to a table entry and inspect one level.
  *  The function returns 0 on lookup success, ENOENT if no match was found
  *  or 1 if the process needs to be continued by calling this again. */
static inline int
lookup_step(const struct rte_lpm6 *lpm, const struct rte_lpm6_tbl_entry *tbl,
		const struct rte_lpm6_tbl_entry **tbl_next, uint8_t *ip,
		uint8_t first_byte, uint8_t *next_hop)
{
	uint32_t tbl8_index, tbl_entry;
	
	/* Take the integer value from the pointer. */
	tbl_entry = *(const uint32_t *) tbl;
	
	/* If it is valid and extended we calculate the new pointer to return. */
	if ((tbl_entry & RTE_LPM6_VALID_EXT_ENTRY_BITMASK) ==
			RTE_LPM6_VALID_EXT_ENTRY_BITMASK) {

		tbl8_index = ip[first_byte - 1] +
				((tbl_entry & RTE_LPM6_TBL8_BITMASK) *
				RTE_LPM6_TBL8_GROUP_NUM_ENTRIES);

		*tbl_next = &lpm->tbl8[tbl8_index];

		return 1;
	} else {
		/* If not extended then we can have a match. */
		*next_hop = (uint8_t)tbl_entry;
		return (tbl_entry & RTE_LPM6_LOOKUP_SUCCESS) ? 0 : -ENOENT;
	}
}

/**< Process a batch of IPv6 packets. Unlike IPv4, we don't do a packet
 *  validity check here (similar to simple_ipv6_fwd_4pkts() in l3fwd */
void process_batch_goto(struct rte_mbuf **pkts, int nb_pkts, int port_id,
                          const struct rte_lpm6 *lpm,
                          struct lcore_port_info *lp_info)
{
	struct ether_hdr *eth_hdr[BATCH_SIZE];
	struct ipv6_hdr *ip6_hdr[BATCH_SIZE];
	const struct rte_lpm6_tbl_entry *tbl[BATCH_SIZE];
	const struct rte_lpm6_tbl_entry *tbl_next[BATCH_SIZE];
	uint32_t tbl24_index[BATCH_SIZE];
	uint8_t next_hop[BATCH_SIZE];
	uint8_t first_byte[BATCH_SIZE];
	int status[BATCH_SIZE];
	uint8_t *dst_addr[BATCH_SIZE];

	int I = 0;			// batch index
	void *batch_rips[BATCH_SIZE];		// goto targets
	int iMask = 0;		// No packet is done yet

	int temp_index;
	for(temp_index = 0; temp_index < BATCH_SIZE; temp_index ++) {
		batch_rips[temp_index] = &&fpp_start;
	}

fpp_start:

        /**< TX boilerplate */
        
        eth_hdr[I] = (struct ether_hdr *) pkts[I]->pkt.data;
        ip6_hdr[I] = (struct ipv6_hdr *) ((char *) eth_hdr[I] + sizeof(struct ether_hdr));
        
        if(I != nb_pkts - 1) {
            rte_prefetch0(pkts[I + 1]->pkt.data);
        }
        
        /**< %%% Code for IPv6 lookup: from rte_lpm6_lookup_nogoto() %%% */
        
        dst_addr[I] = ip6_hdr[I]->dst_addr;
        first_byte[I] = LOOKUP_FIRST_BYTE;
        tbl24_index[I] = (dst_addr[I][0] << BYTES2_SIZE) |
        (dst_addr[I][1] << BYTE_SIZE) | dst_addr[I][2];
        
        /**< Calculate pointer to the first entry to be inspected */
        tbl[I] = &lpm->tbl24[tbl24_index[I]];
        
        do {
            FPP_PSS(tbl[I], fpp_label_1, nb_pkts);
fpp_label_1:

            /**< Continue inspecting next levels until success or failure */
            status[I] = lookup_step(lpm,
                                 tbl[I], &tbl_next[I], dst_addr[I], first_byte[I] ++, &next_hop[I]);
            tbl[I] = tbl_next[I];
        } while (status[I] == 1);
        
        /**< %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
        
        if(unlikely(status[I] < 0)) {
            rte_pktmbuf_free(pkts[I]);
            lp_info[0].nb_lookup_fail_all_ports ++;
        } else {
            /**< TX boilerplate: use the computed next_hop for L2 src and dst. */
            set_mac(eth_hdr[I]->s_addr.addr_bytes, src_mac_arr[next_hop[I]]);
            set_mac(eth_hdr[I]->d_addr.addr_bytes, dst_mac_arr[next_hop[I]]);
            
            send_packet(pkts[I], next_hop[I], lp_info);
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

/**< Process a batch of IPv6 packets. Unlike IPv4, we don't do a packet
  *  validity check here (similar to simple_ipv6_fwd_4pkts() in l3fwd */
void process_batch_nogoto(struct rte_mbuf **pkts, int nb_pkts, int port_id,
	const struct rte_lpm6 *lpm, 
	struct lcore_port_info *lp_info)
{
	int batch_index = 0;

	foreach(batch_index, nb_pkts) {

		/**< TX boilerplate */
		struct ether_hdr *eth_hdr;
		struct ipv6_hdr *ip6_hdr;
		eth_hdr = (struct ether_hdr *) pkts[batch_index]->pkt.data;
		ip6_hdr = (struct ipv6_hdr *) ((char *) eth_hdr + sizeof(struct ether_hdr));

		if(batch_index != nb_pkts - 1) {
			rte_prefetch0(pkts[batch_index + 1]->pkt.data);
		}

		/**< %%% Code for IPv6 lookup: from rte_lpm6_lookup_nogoto() %%% */
		const struct rte_lpm6_tbl_entry *tbl;
		const struct rte_lpm6_tbl_entry *tbl_next;
		uint32_t tbl24_index;
		uint8_t first_byte, next_hop;
		int status;

		uint8_t *dst_addr = ip6_hdr->dst_addr;
		first_byte = LOOKUP_FIRST_BYTE;
		tbl24_index = (dst_addr[0] << BYTES2_SIZE) |
				(dst_addr[1] << BYTE_SIZE) | dst_addr[2];

		/**< Calculate pointer to the first entry to be inspected */
		tbl = &lpm->tbl24[tbl24_index];

		do {
			FPP_EXPENSIVE(tbl);
			/**< Continue inspecting next levels until success or failure */
			status = lookup_step(lpm,
					tbl, &tbl_next, dst_addr, first_byte ++, &next_hop);
			tbl = tbl_next;
		} while (status == 1);

		/**< %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */

		if(unlikely(status < 0)) {
			rte_pktmbuf_free(pkts[batch_index]);
			lp_info[0].nb_lookup_fail_all_ports ++;
		} else {
			/**< TX boilerplate: use the computed next_hop for L2 src and dst. */
			set_mac(eth_hdr->s_addr.addr_bytes, src_mac_arr[next_hop]);
			set_mac(eth_hdr->d_addr.addr_bytes, dst_mac_arr[next_hop]);

			send_packet(pkts[batch_index], next_hop, lp_info);
		}
	}
}

/**< Forward packets on a random port without performing IPv6 lookups.
  *  Enabled by setting PASSTHROUGH = 1 in main.h */
void process_batch_passthrough(struct rte_mbuf **pkts, int nb_pkts, int port_id,
	struct lcore_port_info *lp_info)
{
	int batch_index = 0;

	foreach(batch_index, nb_pkts) {

		/**< TX boilerplate */
		struct ether_hdr *eth_hdr;
		struct ipv6_hdr *ip6_hdr;
		eth_hdr = (struct ether_hdr *) pkts[batch_index]->pkt.data;
		ip6_hdr = (struct ipv6_hdr *) ((char *) eth_hdr + sizeof(struct ether_hdr));

		if(batch_index != nb_pkts - 1) {
			rte_prefetch0(pkts[batch_index + 1]->pkt.data);
		}

		/**< XXX: Assumes ports 0-3 are enabled */
		int next_hop = ip6_hdr->src_addr[0] & 3;

		/**< TX boilerplate: use the computed next_hop for L2 src and dst. */
		set_mac(eth_hdr->s_addr.addr_bytes, src_mac_arr[next_hop]);
		set_mac(eth_hdr->d_addr.addr_bytes, dst_mac_arr[next_hop]);

		send_packet(pkts[batch_index], next_hop, lp_info);
	}
}

void run_server(struct rte_lpm6 *lpm)
{
	int i;

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

#if PASSTHROUGH == 1
		process_batch_passthrough(rx_pkts_burst, nb_rx_new, port_id,
			lp_info);
#else
	#if GOTO == 1
		process_batch_goto(rx_pkts_burst, nb_rx_new, port_id,
			lpm, lp_info);
	#else
		process_batch_nogoto(rx_pkts_burst, nb_rx_new, port_id,
			lpm, lp_info);
	#endif
#endif
		
		/**< STAT PRINTING */
		if (unlikely(lp_info[0].nb_tx_all_ports >= 10000000)) {
			tput_tsc[1] = rte_rdtsc();
			double nanoseconds = S_FAC * (tput_tsc[1] - tput_tsc[0]);
			double seconds = nanoseconds / GHZ_CPS;
			tput_tsc[0] = tput_tsc[1];

			red_printf("Lcore %d, total: %f, fail: %d\n", lcore_id, 
				lp_info[0].nb_tx_all_ports / seconds,
				lp_info[0].nb_lookup_fail_all_ports);

			/**< Reset all-port stats in case port 0 is disabled */
			lp_info[0].nb_tx_all_ports = 0;
			lp_info[0].nb_lookup_fail_all_ports = 0;
			
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
