#include "main.h"
#include "city.h"

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

/**< Try to find a match for the 1st component of this name in the hash
  *  table. This function gets called when the clever matching trick of
  *  starting from 2nd component matches fails. */
int lookup_one_component(char *name, struct ndn_bucket *ht)
{
	int c_i, i;	/**< URL char iterator and slot iterator */
	int bkt_num, bkt_1, bkt_2;

	for(c_i = 0; c_i < NDN_MAX_NAME_LENGTH; c_i ++) {
		if(name[c_i] == '/') {
			break;
		}
	}

	/**< c_i is now at the boundary of the 1st component */
	uint64_t prefix_hash = CityHash64(name, c_i + 1);
	uint16_t tag = prefix_hash >> 48;

	struct ndn_slot *slots;

	/**< name[0] -> name[c_i] is a prefix of length c_i + 1 */
	for(bkt_num = 1; bkt_num <= 2; bkt_num ++) {
		if(bkt_num == 1) {
			bkt_1 = prefix_hash & NDN_NUM_BKT_;
			slots = ht[bkt_1].slots;
		} else {
			bkt_2 = (bkt_1 ^ CityHash64((char *) &tag, 2)) & NDN_NUM_BKT_;
			slots = ht[bkt_2].slots;
		}

		/**< Now, "slots" points to an ndn_bucket. Find a valid slot
		  *  that contains the same hash. */
		for(i = 0; i < NDN_NUM_SLOTS; i ++) {
			int8_t _dst_port = slots[i].dst_port;
			uint64_t _hash = slots[i].cityhash;

			if(_dst_port >= 0 && _hash == prefix_hash) {

				/**< As we're only matching this component, we're done! */
				return slots[i].dst_port;
			}
		}
	}

	/**< No match even for the 1st component? */
	return -1;

}

void process_batch_goto(struct rte_mbuf **pkts, int nb_pkts,
                          struct ndn_bucket *ht,
                          struct lcore_port_info *lp_info, int port_id,
                          struct mac_ints *mac_ints_arr)
{
	struct ether_hdr *eth_hdr[BATCH_SIZE];
	char *name[BATCH_SIZE];
	char *data_ptr[BATCH_SIZE];
	int fwd_port[BATCH_SIZE];
	int i[BATCH_SIZE];
	int c_i[BATCH_SIZE];
	int bkt_2[BATCH_SIZE];
	int bkt_1[BATCH_SIZE];
	int bkt_num[BATCH_SIZE];
	int terminate[BATCH_SIZE];
	int prefix_match_found[BATCH_SIZE];
	uint64_t prefix_hash[BATCH_SIZE];
	uint16_t tag[BATCH_SIZE];
	struct ndn_slot *slots[BATCH_SIZE];
	int8_t _dst_port[BATCH_SIZE];
	uint64_t _hash[BATCH_SIZE];
	int *mac_ints_dst[BATCH_SIZE];

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
        data_ptr[I] = (char *) pkts[I]->pkt.data;
        name[I] = data_ptr[I] + 36 + sizeof(int) + sizeof(LL);
        
         /**< URL char iterator and slot iterator */
        
        terminate[I] = 0;          /**< Stop processing this URL? */
        prefix_match_found[I] = 0; /**< Stop this hash-table lookup ? */
        
        /**< Completely ignore 1-component matches */
        for(c_i[I] = 0; name[I][c_i[I]] != 0; c_i[I] ++) {
            if(name[I][c_i[I]] == '/') {
                break;
            }
        }
        c_i[I] ++;
        
        for(; name[I][c_i[I]] != 0; c_i[I] ++) {
            if(name[I][c_i[I]] != '/') {
                continue;
            }
            
            /**< c_i is now at the boundary of a component longer than the 1st */
            prefix_hash[I] = CityHash64(name[I], c_i[I] + 1);
            tag[I] = prefix_hash[I] >> 48;
            
            /**< name[0] -> name[c_i] is a prefix of length c_i + 1 */
            for(bkt_num[I] = 1; bkt_num[I] <= 2; bkt_num[I] ++) {
                if(bkt_num[I] == 1) {
                    bkt_1[I] = prefix_hash[I] & NDN_NUM_BKT_;
                    FPP_PSS(&ht[bkt_1[I]], fpp_label_1, nb_pkts);
fpp_label_1:

                    slots[I] = ht[bkt_1[I]].slots;
                } else {
                    bkt_2[I] = (bkt_1[I] ^ CityHash64((char *) &tag[I], 2)) & NDN_NUM_BKT_;
                    FPP_PSS(&ht[bkt_2[I]], fpp_label_2, nb_pkts);
fpp_label_2:

                    slots[I] = ht[bkt_2[I]].slots;
                }
                
                /**< Now, "slots" points to an ndn_bucket. Find a valid slot
                 *  that contains the same hash. */
                for(i[I] = 0; i[I] < NDN_NUM_SLOTS; i[I] ++) {
                    _dst_port[I] = slots[I][i[I]].dst_port;
                    _hash[I] = slots[I][i[I]].cityhash;
                    
                    if(_dst_port[I] >= 0 && _hash[I] == prefix_hash[I]) {
                        
                        /**< Record the dst port: this may get overwritten by
                         *  longer prefix matches later */
                        fwd_port[I] = slots[I][i[I]].dst_port;
                        
                        if(slots[I][i[I]].is_terminal == 1) {
                            /**< A terminal FIB entry: we're done! */
                            terminate[I] = 1;
                        }
                        
                        prefix_match_found[I] = 1;
                        break;
                    }
                }
                
                /**< Stop the hash-table lookup for name[0 ... c_i] */
                if(prefix_match_found[I] == 1) {
                    break;
                }
            }
            
            /**< Stop processing the name if we found a terminal FIB entry */
            if(terminate[I] == 1) {
                break;
            }
        }   /**< Loop over URL characters ends here */
        
		/**< We failed to match with prefixes that match >=2 components of this
		  *  name. Try matching only the 1st component now. */
		if(fwd_port[I] == -1) {
			fwd_port[I] = lookup_one_component(name[I], ht);
		}

        /**< Count failed packets and transmit */
        if(fwd_port[I] == -1) {
            lp_info[port_id].nb_loookup_fail ++;
            rte_pktmbuf_free(pkts[I]);
        } else {
			mac_ints_dst[I] = (int *) eth_hdr[I];
			mac_ints_dst[I][0] = mac_ints_arr[fwd_port[I]].chunk[0];
			mac_ints_dst[I][1] = mac_ints_arr[fwd_port[I]].chunk[1];
			mac_ints_dst[I][2] = mac_ints_arr[fwd_port[I]].chunk[2];

			/**< Garble dst MAC to reduce RX load on clients */
			eth_hdr[I]->d_addr.addr_bytes[0] += bkt_1[I] & 0xff;

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
	struct ndn_bucket *ht,
	struct lcore_port_info *lp_info, int port_id,
	struct mac_ints *mac_ints_arr)
{

	int batch_index = 0;		/**< Don't make global!! */

	foreach(batch_index, nb_pkts) {
		struct ether_hdr *eth_hdr;
		char *data_ptr, *name;

		int fwd_port = -1;

		if(batch_index != nb_pkts - 1) {
			rte_prefetch0(pkts[batch_index + 1]->pkt.data);
		}

		eth_hdr = (struct ether_hdr *) pkts[batch_index]->pkt.data;
		data_ptr = (char *) pkts[batch_index]->pkt.data;
		name = data_ptr + 36 + sizeof(int) + sizeof(LL);

		int c_i, i;	/**< URL char iterator and slot iterator */
		int bkt_num, bkt_1 = 0, bkt_2;

		int terminate = 0;			/**< Stop processing this URL? */
		int prefix_match_found = 0;	/**< Stop this hash-table lookup ? */

		/**< Completely ignore 1-component matches */		
		for(c_i = 0; name[c_i] != 0; c_i ++) {
			if(name[c_i] == '/') {
				break;
			}
		}
		c_i ++;

		for(; name[c_i] != 0; c_i ++) {
			if(name[c_i] != '/') {
				continue;
			}

			/**< c_i is now at the boundary of a component longer than the 1st */
			uint64_t prefix_hash = CityHash64(name, c_i + 1);
			uint16_t tag = prefix_hash >> 48;

			struct ndn_slot *slots;

			/**< name[0] -> name[c_i] is a prefix of length c_i + 1 */
			for(bkt_num = 1; bkt_num <= 2; bkt_num ++) {
				if(bkt_num == 1) {
					bkt_1 = prefix_hash & NDN_NUM_BKT_;
					FPP_EXPENSIVE(&ht[bkt_1]);
					slots = ht[bkt_1].slots;
				} else {
					bkt_2 = (bkt_1 ^ CityHash64((char *) &tag, 2)) & NDN_NUM_BKT_;
					FPP_EXPENSIVE(&ht[bkt_2]);
					slots = ht[bkt_2].slots;
				}

				/**< Now, "slots" points to an ndn_bucket. Find a valid slot
				  *  that contains the same hash. */
				for(i = 0; i < NDN_NUM_SLOTS; i ++) {
					int8_t _dst_port = slots[i].dst_port;
					uint64_t _hash = slots[i].cityhash;

					if(_dst_port >= 0 && _hash == prefix_hash) {

						/**< Record the dst port: this may get overwritten by
						  *  longer prefix matches later */
						fwd_port = slots[i].dst_port;

						if(slots[i].is_terminal == 1) {
							/**< A terminal FIB entry: we're done! */
							terminate = 1;
						}

						prefix_match_found = 1;
						break;
					}
				}

				/**< Stop the hash-table lookup for name[0 ... c_i] */
				if(prefix_match_found == 1) {
					break;
				}
			}

			/**< Stop processing the name if we found a terminal FIB entry */
			if(terminate == 1) {
				break;
			}
		}	/**< Loop over URL characters ends here */

		/**< We failed to match with prefixes that contain 2 or more
		  *  components. Try matching the 1st component of this name now */
		if(fwd_port == -1) {
			fwd_port = lookup_one_component(name, ht);
		}

		/**< Count failed packets and transmit */
		if(fwd_port == -1) {
			lp_info[port_id].nb_loookup_fail ++;
			rte_pktmbuf_free(pkts[batch_index]);
		} else {
			int *mac_ints_dst = (int *) eth_hdr;
			mac_ints_dst[0] = mac_ints_arr[fwd_port].chunk[0];
			mac_ints_dst[1] = mac_ints_arr[fwd_port].chunk[1];
			mac_ints_dst[2] = mac_ints_arr[fwd_port].chunk[2];

			/**< Garble dst MAC to reduce RX load on clients */
			eth_hdr->d_addr.addr_bytes[0] += bkt_1 & 0xff;
			send_packet(pkts[batch_index], fwd_port, lp_info);
		}
	}
}

void run_server(struct ndn_bucket *ht)
{
	int i;

	int lcore_id = rte_lcore_id();
	int socket_id = rte_lcore_to_socket_id(lcore_id);
	assert(socket_id == 0);

	int queue_id = get_lcore_rank(lcore_id, socket_id);
	printf("Server on lcore %d. Queue Id = %d\n", lcore_id, queue_id);

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

	/**< Init measurement variables */
	LL tput_tsc[2], brst_sz_msr[4];
	tput_tsc[0] = rte_rdtsc();
	memset(brst_sz_msr, 0, 4 * sizeof(LL));

	while (1) {
		int port_id = port_arr[port_index];	/**< Port for this iteration */
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

#if GOTO == 1
		process_batch_goto(rx_pkts_burst, 
			nb_rx_new, ht, lp_info, port_id, mac_ints_arr);
#else
		process_batch_nogoto(rx_pkts_burst,
			nb_rx_new, ht, lp_info, port_id, mac_ints_arr);
#endif
		
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
					printf("\tLcore: %d, port %d: S: %f, F: %f\n", lcore_id, i,
						lp_info[i].nb_tx / seconds, 
						lp_info[i].nb_loookup_fail / seconds);
				}

				/**< Do not reset the nb_buf counter */
				lp_info[i].nb_tx = 0;
				lp_info[i].nb_loookup_fail = 0;

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
