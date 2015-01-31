void process_batch(int *key_lo)
{
    foreach(batch_index, BATCH_SIZE) {
        int i, bkt_1, bkt_2, success = 0;
        int key = key_lo[batch_index];
        
        /** < Try the first bucket */
        bkt_1 = hash(key) & NUM_BKT_;
        FPP_EXPENSIVE(&ht_index[bkt_1]);
        
        for(i = 0; i < 8; i ++) {
            if(ht_index[bkt_1].slot[i].key == key) {
                sum += ht_index[bkt_1].slot[i].value;
                succ_1 ++;
                success = 1;
                break;
            }
        }
        
        if(success == 0) {
            bkt_2 = hash(bkt_1) & NUM_BKT_;
            FPP_EXPENSIVE(&ht_index[bkt_2]);
            
            for(i = 0; i < 8; i ++) {
                if(ht_index[bkt_2].slot[i].key == key) {
                    sum += ht_index[bkt_2].slot[i].value;
                    succ_2 ++;
                    success = 1;
                    break;
                }
            }
        }
        
        if(success == 0) {
            fail ++;
        }
    }
}
