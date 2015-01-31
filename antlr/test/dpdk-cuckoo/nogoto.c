void process_batch_nogoto(struct rte_mbuf **pkts, int nb_pkts,
                          uint64_t *rss_seed, struct cuckoo_bucket *ht_index,
                          struct lcore_port_info *lp_info, int port_id)
{
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
		dst_mac = get_mac(eth_hdr->d_addr.addr_bytes);
        
		eth_hdr->ether_type = htons(ETHER_TYPE_IPv4);
        
		// These 3 fields of ip_hdr are required for RSS
		ip_hdr->src_addr = fastrand(rss_seed);
		ip_hdr->dst_addr = fastrand(rss_seed);
		ip_hdr->version_ihl = 0x40 | 0x05;
        
		pkts[batch_index]->pkt.nb_segs = 1;
		pkts[batch_index]->pkt.pkt_len = 60;
		pkts[batch_index]->pkt.data_len = 60;
        
		bkt_1 = CityHash32(dst_mac_ptr, 6) & NUM_BKT_;
		FPP_EXPENSIVE(&ht_index[bkt_1]);
        
		for(i = 0; i < 8; i ++) {
			if(SLOT_TO_MAC(ht_index[bkt_1].slot[i]) == dst_mac) {
				fwd_port = SLOT_TO_PORT(ht_index[bkt_1].slot[i]);
				break;
			}
		}
        
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
