// Process BATCH_SIZE pkts starting from lo
int process_pkts_in_batch(int *pkt_lo)
{
	// Like a foreach loop
	foreach(batch_index, BATCH_SIZE) {
		
		int i;
		int jumper = pkt_lo[batch_index];
			
		for(i = 0; i < DEPTH; i++) {
			int *arr = cache[jumper].slot_arr;
			int j, best_j = 0;

			int max_diff = ABS(arr[0] - jumper) % 8;

			for(j = 1; j < SLOTS_PER_BKT; j ++) {
				if(ABS(arr[j] - jumper) % 8 > max_diff) {
					max_diff = ABS(arr[j] - jumper) % 8;
					best_j = j;
				}
			}
			
			jumper = arr[best_j];
			if(jumper % 16 == 0) {		// GCC will optimize this
				break;
			}
		}

		sum += jumper;
	}
}
