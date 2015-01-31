void process_batch(struct node *nodes)
{
    foreach(batch_index, BATCH_SIZE) {
        int i, next_nbh;
        struct node *cur_node = &nodes[batch_index];
        
        for(i = 0; i < STEPS; i ++) {
            FPP_EXPENSIVE(cur_node);
            sum += cur_node->id;
            
            /** < Compute the next neighbor */
            next_nbh = -1;
            while(next_nbh < 0) {
                next_nbh = rand() % 7;
            }
            
            cur_node = (struct node *) nodes[batch_index].neighbors[next_nbh];
        }
        
    }   
}
