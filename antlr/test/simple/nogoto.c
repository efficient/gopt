// Process BATCH_SIZE pkts starting from lo
int process_pkts_in_batch(int *pkt_lo)
{
	// Like a foreach loop
	foreach(batch_index, BATCH_SIZE) {

		int mem_addr = hash(pkt_lo[batch_index]) & LOG_CAP_;
		PREFETCH(&ht_log[mem_addr]);
		sum += ht_log[mem_addr];
	}
}

