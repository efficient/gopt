void process_pkts_in_batch(LL *pkt_lo)
{
    foreach(batch_index, BATCH_SIZE) {
        LL key_hash = hash(pkt_lo[batch_index]);
        
        int key_tag = HASH_TO_TAG(key_hash);
        int ht_bucket = HASH_TO_BUCKET(key_hash);
        
        FPP_EXPENSIVE(&ht_index[ht_bucket]);
        LL *slots = ht_index[ht_bucket].slots;
        
        int i, found = 0;
        
        for(i = 0; i < SLOTS_PER_BKT; i ++) {
            
            // Tag matched
            if(SLOT_TO_TAG(slots[i]) == key_tag &&
               SLOT_TO_LOG_I(slots[i]) != INVALID_KV_I) {
                int log_i = SLOT_TO_LOG_I(slots[i]);
                FPP_EXPENSIVE(&ht_log[log_i]);
                
                // Log entry also matches
                if(ht_log[log_i].key == pkt_lo[batch_index]) {
                    found = 1;
                    succ ++;
                    sum += (int) ht_log[log_i].value;
                    break;
                } else {
                    fail_1 ++;
                }
            }
        }
        
        if(found == 0) {
            fail_2 ++; 
        }   
    }   
}