void process_batch(struct aho_state *dfa, struct pkt *test_pkts)
{
    foreach(batch_index, BATCH_SIZE) {
        int j;
        int state = 0;
        
        for(j = 0; j < PKT_SIZE; j ++) {
            int inp = test_pkts[batch_index].content[j];
            state = dfa[state].G[inp];
            if(j != PKT_SIZE - 1) {
                FPP_EXPENSIVE(&dfa[state].G[test_pkts[batch_index].content[j + 1]]);
            }
        }
        
        final_state_sum += state;
    }   
}
