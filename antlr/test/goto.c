// Process BATCH_SIZE pkts starting from lo
int process_pkts_in_batch(int *pkt_lo)
{
	int I = 0;			// batch index
	void *batch_rips[BATCH_SIZE];		// goto targets
	int iMask = 0;		// No packet is done yet

	int i[BATCH_SIZE];
	int jumper[BATCH_SIZE];
	int *arr[BATCH_SIZE];
	int j[BATCH_SIZE];
	int best_j[BATCH_SIZE];
	int max_diff[BATCH_SIZE];

	int temp_index;
	for(temp_index = 0; temp_index < BATCH_SIZE; temp_index ++) {
		batch_rips[temp_index] = &&label_0;
	}

label_0:
	
	jumper[I] = pkt_lo[I];
			
	for(i[I] = 0; i[I] < DEPTH; i[I] ++) {
		PSS(&cache[jumper[I]], label_1);
			
label_1:
		arr[I] = cache[jumper[I]].slot_arr;
		best_j[I] = 0;

		max_diff[I] = ABS(arr[I][0] - jumper[I]) % 8;

		for(j[I] = 1; j[I] < SLOTS_PER_BKT; j[I] ++) {
			if(ABS(arr[I][j[I]] - jumper[I]) % 8 > max_diff[I]) {
				max_diff[I] = ABS(arr[I][j[I]] - jumper[I]) % 8;
				best_j[I] = j[I];
			}
		}
		
		jumper[I] = arr[I][best_j[I]];
		if(jumper[I] % 16 == 0) {
			break;
		}
	}

	sum += jumper[I];

end:
	batch_rips[I] = &&end;
	iMask = SET(iMask, I);
	if(iMask == (1 << BATCH_SIZE) - 1) {
		return;
	}
	I = (I + 1) & BATCH_SIZE_;
	goto *batch_rips[I];
}
