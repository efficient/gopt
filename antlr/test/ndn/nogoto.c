void process_batch_nogoto(struct rte_mbuf **pkts, int nb_pkts,
                          struct ndn_bucket *ht,
                          struct lcore_port_info *lp_info, int port_id)
{
    foreach(batch_index, nb_pkts) {
        struct ether_hdr *eth_hdr;
        char *data_ptr, *name;
        
        int fwd_port = -1;
        
        if(batch_index != nb_pkts - 1) {
            rte_prefetch0(pkts[batch_index + 1]->pkt.data);
        }
        
        eth_hdr = (struct ether_hdr *) pkts[batch_index]->pkt.data;
        data_ptr = (char *) pkts[batch_index]->pkt.data;
        name = data_ptr + HDR_SIZE + sizeof(int) + sizeof(LL);
        
        int c_i, i; /**< URL char iterator and slot iterator */
        int bkt_num, bkt_1, bkt_2;
        
        int terminate = 0;          /**< Stop processing this URL? */
        int prefix_match_found = 0; /**< Stop this hash-table lookup ? */
        
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
            uint64_t prefix_hash = CityHash64WithSeed(name, c_i + 1, NDN_SEED);
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
                 *  with a matching tag. */
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
        }   /**< Loop over URL characters ends here */
        
        /**< Count failed packets and transmit */
        if(fwd_port == -1) {
            lp_info[port_id].nb_tx_fail ++;
            rte_pktmbuf_free(pkts[batch_index]);
        } else {
            set_mac(eth_hdr->d_addr.addr_bytes, dst_mac_arr[fwd_port]);
            
            /**< Reduce RX load on client: If the client sent a bad
             *  src MAC address, garble dst MAC address */
            if(eth_hdr->s_addr.addr_bytes[0] == 0xef) {
                eth_hdr->d_addr.addr_bytes[0] ++;
            }
            set_mac(eth_hdr->s_addr.addr_bytes, src_mac_arr[port_id]);
            send_packet(pkts[batch_index], fwd_port, lp_info);
        }  
    }  
}
