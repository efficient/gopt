void process_batch_nogoto(struct rte_mbuf **pkts, int nb_pkts, int port_id,
                          const struct rte_lpm *lpm,
                          struct lcore_port_info *lp_info)
{
    foreach(batch_index, nb_pkts) {
        
        /**< Boilerplate for TX pkt */
        struct ether_hdr *eth_hdr;
        struct ipv4_hdr *ip_hdr;
        
        uint32_t dst_ip;
        int dst_port;
        
        if(batch_index != nb_pkts - 1) {
            rte_prefetch0(pkts[batch_index + 1]->pkt.data);
        }
        
        eth_hdr = (struct ether_hdr *) pkts[batch_index]->pkt.data;
        ip_hdr = (struct ipv4_hdr *) ((char *) eth_hdr + sizeof(struct ether_hdr));
        
        if(is_valid_ipv4_pkt(ip_hdr, pkts[batch_index]->pkt.pkt_len) < 0) {
            rte_pktmbuf_free(pkts[batch_index]);
            continue;
        }
        
        set_mac(eth_hdr->s_addr.addr_bytes, src_mac_arr[port_id]);
        
        ip_hdr->time_to_live --;
        ip_hdr->hdr_checksum ++;
        
        dst_ip = ip_hdr->dst_addr;
        
        /**< Copied code from DPDK's rte_lpm.h */
        /**%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
        unsigned tbl24_index = (dst_ip >> 8);
        uint16_t tbl_entry;
        
        /**< Copy tbl24 entry */
        FPP_EXPENSIVE(&lpm->tbl24[tbl_index]);
        tbl_entry = *(const uint16_t *) &lpm->tbl24[tbl24_index];
        
        /**< Copy tbl8 entry (only if needed) */
        if (unlikely((tbl_entry & RTE_LPM_VALID_EXT_ENTRY_BITMASK) ==
                     RTE_LPM_VALID_EXT_ENTRY_BITMASK)) {
            
            unsigned tbl8_index = (uint8_t) dst_ip +
            ((uint8_t) tbl_entry * RTE_LPM_TBL8_GROUP_NUM_ENTRIES);
            
            tbl_entry = *(const uint16_t *)&lpm->tbl8[tbl8_index];
        }
        
        dst_port = (uint8_t) tbl_entry;
        /**%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
        
        /**< Use the looked-up port to determine dst MAC */
        set_mac(eth_hdr->d_addr.addr_bytes, dst_mac_arr[dst_port]);
        
        send_packet(pkts[batch_index], dst_port, lp_info);
    }
}