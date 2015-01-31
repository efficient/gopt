void rte_lpm6_lookup_nogoto(const struct rte_lpm6 *lpm,
                            uint8_t ips[][RTE_LPM6_IPV6_ADDR_SIZE],
                            int16_t *next_hops, unsigned n)
{
    foreach(batch_index, n) {
        const struct rte_lpm6_tbl_entry *tbl;
        const struct rte_lpm6_tbl_entry *tbl_next;
        uint32_t tbl24_index;
        uint8_t first_byte, next_hop;
        int status;
        
        first_byte = LOOKUP_FIRST_BYTE;
        tbl24_index = (ips[batch_index][0] << BYTES2_SIZE) |
        (ips[batch_index][1] << BYTE_SIZE) | ips[batch_index][2];
        
        /* Calculate pointer to the first entry to be inspected */
        tbl = &lpm->tbl24[tbl24_index];
        
        do {
            FPP_EXPENSIVE(tbl);
            /* Continue inspecting following levels until success or failure */
            status = lookup_step(lpm, tbl, &tbl_next, ips[batch_index], first_byte++,
                                 &next_hop);
            tbl = tbl_next;
        } while (status == 1);
        
        if (status < 0)
            next_hops[batch_index] = -1;
        else
            next_hops[batch_index] = next_hop;
    }   
}