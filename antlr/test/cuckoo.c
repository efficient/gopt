static int _cuckoopath_search(cuckoo_hashtable_t *h,
                              size_t depth_start,
                              size_t *cp_index)
{
    int   depth = depth_start;
    
    while ((h->kick_count < MAX_CUCKOO_COUNT) && (depth >= 0) && (depth < MAX_CUCKOO_COUNT - 1))
    {
        cuckoo_record_t **curr = h->cuckoo_path + depth;
        cuckoo_record_t **next = h->cuckoo_path + depth + 1;
        
        size_t idx;
        for (idx = 0; idx < NUM_CUCKOO_PATH; idx++)
        {
            size_t i;
            size_t j;
            
            i = curr->buckets[idx];
            for (j = 0; j < bucketsize; j++)
            {
                if (_is_slot_empty(h, i, j))
                {
                    curr->slots[idx] = j;
                    *cp_index = idx;
                    return depth;
                }
            }
            
            j = rand() % bucketsize;
            curr->slots[idx] = j;
            curr->keys[idx] = TABLE_KEY(h, i, j);
            
            uint32_t hv = _hashed_key(curr->keys[idx]);
            next->buckets[idx] = _alt_index(h, hv, i);
        }
        
        h->kick_count += NUM_CUCKOO_PATH;
        depth++;
    }    
    
    return -1;
}
