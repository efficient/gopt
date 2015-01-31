/**< Process a batch of IPv6 packets. Unlike IPv4, we don't do a packet
 *  validity check here (similar to simple_ipv6_fwd_4pkts() in l3fwd */
void process_batch_nogoto(struct rte_mbuf **pkts, int nb_pkts, int port_id,
                          const struct rte_lpm6 *lpm,
                          struct lcore_port_info *lp_info)
{    
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
