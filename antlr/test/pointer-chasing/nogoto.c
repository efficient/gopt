// Process BATCH_SIZE pkts starting from lo
int process_pkts_in_batch(int *pkt_lo)
{
    // Like a foreach loop
    foreach(batch_index, BATCH_SIZE) {
        
        int i;
        int jumper = pkt_lo[batch_index];
        
        for(i = 0; i < DEPTH; i++) {
            FPP_EXPENSIVE(&ht_log[jumper]);
            jumper = ht_log[jumper];
        }
        
        sum += jumper;
    }   
}
